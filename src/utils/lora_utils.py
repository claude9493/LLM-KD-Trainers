import torch


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


# vllm+lora
# ref: https://github.com/vllm-project/vllm/issues/182
def transpose(weight, fan_in_fan_out):
    return weight.T if fan_in_fan_out else weight


def lora_reassign_weights(
    model, state_dict, r, lora_alpha, fan_in_fan_out=False, merge=True
):
    is_merged = getattr(model, "is_merged", False)
    assert (
        is_merged != merge
    ), f"{is_merged} != {merge}: if is_merged, then must be unmerge; if not is_merged, then must merge"
    named_params = [(n, p) for n, p in model.named_parameters()]
    scaling = lora_alpha / r
    print(f"Lora configs: alpha={lora_alpha}, r={r}, scaling={scaling}")
    state_dict = {k.replace("base_model.model.", ""): v for k, v in state_dict.items()}
    replaced = set()
    merged_names = {
        # these are projector weights that got combined into single matrix in vllm
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    non_merged_names = ["o_proj", "down_proj"]
    for name, param in named_params:
        param.requires_grad = False
        if "_proj.weight" not in name:
            continue
        for wn, wn_series in merged_names.items():
            if name.endswith(f"{wn}.weight"):
                for stride_id, att_weight_name in enumerate(wn_series):
                    lora_a = name.replace(
                        f"{wn}.weight", f"{att_weight_name}.lora_A.weight"
                    )
                    lora_b = name.replace(
                        f"{wn}.weight", f"{att_weight_name}.lora_B.weight"
                    )
                    shard_size = param.shape[0] // len(wn_series)
                    if lora_a in state_dict:
                        assert lora_b in state_dict, f"{lora_b} not in state_dict"
                        assert (
                            state_dict[lora_b].shape[1] == r
                        ), f"{r=} != {state_dict[lora_b].shape}"
                        matrix = (
                            transpose(
                                state_dict[lora_b] @ state_dict[lora_a], fan_in_fan_out
                            )
                            * scaling
                        )
                        assert (
                            param.data[
                                shard_size * stride_id : shard_size * (stride_id + 1)
                            ].shape
                            == matrix.shape
                        )
                        if merge:
                            param.data[
                                shard_size * stride_id : shard_size * (stride_id + 1)
                            ] += matrix.to(param.device)
                        else:
                            param.data[
                                shard_size * stride_id : shard_size * (stride_id + 1)
                            ] -= matrix.to(param.device)
                        replaced.add(lora_a)
                        replaced.add(lora_b)
        for wn in non_merged_names:
            if name.endswith(f"{wn}.weight"):
                lora_a = name.replace(f"{wn}.weight", f"{wn}.lora_A.weight")
                lora_b = name.replace(f"{wn}.weight", f"{wn}.lora_B.weight")
                if lora_a in state_dict:
                    assert lora_b in state_dict
                    matrix = (
                        transpose(
                            state_dict[lora_b] @ state_dict[lora_a], fan_in_fan_out
                        )
                        * scaling
                    )
                    assert (
                        param.data.shape == matrix.shape
                    ), f"invalid shape: {name} {param.data.shape} != {matrix.shape}"
                    if merge:
                        param.data += matrix.to(param.device)
                    else:
                        param.data -= matrix.to(param.device)
                    replaced.add(lora_a)
                    replaced.add(lora_b)
    no_replaced = [k for k in state_dict.keys() if k not in replaced]
    assert (
        len(no_replaced) == 0
    ), f"some lora states not loaded, check again!: {no_replaced}"
    model.is_merged = merge


def lora_merge_unmerge_state_dict(llm, state_dict, peft_config, merge=True):
    # merge lora states to weights
    for worker in llm.llm_engine.workers:
        lora_reassign_weights(
            worker.model,
            state_dict,
            r=peft_config.r,
            lora_alpha=peft_config.lora_alpha,
            fan_in_fan_out=peft_config.fan_in_fan_out,
            merge=merge,
        )
