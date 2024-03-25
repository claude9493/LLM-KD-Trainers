from copy import deepcopy
from loguru import logger
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import tensor_parallel as tp

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import Seq2SeqTrainer, TrainerCallback, GenerationConfig

from transformers.trainer import (
    is_sagemaker_mp_enabled, 
    is_peft_available, 
    unwrap_model, 
    is_apex_available,
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
)

from .kd_arguments import *
from .losses import forward_kl, reverse_kl

if is_apex_available():
    from apex import amp
    
if is_peft_available():
    from peft import PeftModel

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION
    from transformers.trainer import smp_forward_backward

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from .trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

if TYPE_CHECKING:
    from transformers.data.data_collator import DataCollator
    from transformers.modeling_utils import PreTrainedModel
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    from transformers.trainer_callback import TrainerCallback
    from transformers.trainer_utils import EvalPrediction, PredictionOutput
    from transformers.training_args import TrainingArguments

def kld_loss(input, target, mask, reduction="none"):
    # lambda input, target, mask: (F.kl_div(input.log()*mask, target*mask, reduction="none") / mask.sum(1, keepdim=True)).view(len(input),-1).sum(-1).mean()
    kld = F.kl_div((input*mask).nan_to_num(0,0,0), (target*mask).nan_to_num(0,0,0), reduction="none")
    if reduction == "mean":
        return (kld / mask.sum(1, keepdim=True)).view(len(input),-1).nansum(-1).mean()
    elif reduction == "none":
        return kld

class KDLoggingCallback(TrainerCallback):
    
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer
    
    def on_step_end(self, args, state, control, **kwargs):
        if control.should_log:
            self._trainer.log(self._trainer.loss_dict)
        self._trainer.loss_dict = dict()



class Seq2SeqKDTrainer(Seq2SeqTrainer):
    def __init__(
        self,
        model: Union["PreTrainedModel", nn.Module] = None,
        teacher_model: Union["PreTrainedModel", nn.Module] = None,
        args: "TrainingArguments" = None,
        kd_args: "Seq2SeqKDArguments" = None,
        data_collator: Optional["DataCollator"] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        model_init: Optional[Callable[[], "PreTrainedModel"]] = None,
        compute_metrics: Optional[Callable[["EvalPrediction"], Dict]] = None,
        callbacks: Optional[List["TrainerCallback"]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        tensor_parallel: bool = False
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        
        # Load Teacher model
        self.teacher_model = teacher_model
        self.tensor_parallel = tensor_parallel

        if tensor_parallel:
            self.model = tp.tensor_parallel(self.model)
            self.teacher_model = tp.tensor_parallel(self.teacher_model)
            logger.debug("Student and teacher models are tensor-parallized.")
        else:
            self._move_model_to_device(self.teacher_model, args.device)
            self._move_model_to_device(self.model, args.device)
            logger.debug(f"Models are moved to device {args.device}")
        
        self.kd_args = kd_args
        self.loss_dict = dict()
        

       
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        transformers==4.36.2 !!!

        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        if self.tensor_parallel and type(model) != tp.TensorParallelPreTrainedModel:
            model = tp.tensor_parallel(model)
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None  # Default case: no label smoothing
        
        outputs = model(**inputs)
        
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if is_peft_available() and isinstance(model, PeftModel):
                model_name = unwrap_model(model.base_model)._get_name()
            else:
                model_name = unwrap_model(model)._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            loss_sft = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        
        if self.control.should_evaluate:
            # Return only loss with labels when evaluate.
            self.loss_dict = dict(
                loss=round(loss_sft.mean().item(), 4)
            )
            return (loss_sft, outputs) if return_outputs else loss_sft
        
        loss_kd = self.compute_kd_loss(model, inputs, outputs)
        kd_ratio = self.kd_args.kd_ratio
        loss = (1-kd_ratio)* loss_sft + kd_ratio * loss_kd
        self.loss_dict = dict(
            loss=round(loss.mean().item(), 4),
            loss_sft=round(loss_sft.mean().item(), 4),
            loss_kd=round(loss_kd.mean().item(), 4)
        )
        return (loss, outputs) if return_outputs else loss
    
    def compute_kd_loss(self, model, inputs, outputs=None):
        assert outputs, "Give me student model's outputs."
        tmpt = self.kd_args.kd_temperature
        # Train on output only, mask input and padding tokens.
        loss_mask = torch.where(((inputs['labels'] < 0) | (inputs['labels']==self.tokenizer.pad_token_id)), 0, 1).unsqueeze(-1)        
        with torch.no_grad():
            logits_t = self.teacher_model(**inputs)["logits"]
            
        # input, target = outputs["logits"], logits_t
        
        # if self.kd_args.reverse_kld:
        #     input, target = target, input
            
        # input = F.log_softmax(input/tmpt, dim=-1)  # , dtype=torch.float32)
        # target = F.softmax(target/tmpt, dim=-1)  # , dtype=torch.float32)
        
        # loss_kd = kld_loss(input, target, loss_mask, reduction="mean") * tmpt**2
        
        if self.kd_args.reverse_kld:
            loss_kd = reverse_kl(outputs["logits"], logits_t, mask=loss_mask)
        else:
            loss_kd = forward_kl(outputs["logits"], logits_t, mask=loss_mask)
        return loss_kd
    
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        if self.tensor_parallel and type(model) != tp.TensorParallelPreTrainedModel:
            model = tp.tensor_parallel(model)
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)  #TODO

    def _wrap_model(self, model, training=True, dataloader=None):
        # Overwrite the _wrap_model method
        if self.tensor_parallel:
            return model
        else:
            return super()._wrap_model(model, training, dataloader)
