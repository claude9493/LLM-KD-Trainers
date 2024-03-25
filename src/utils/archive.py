import os
import shutil
from pathlib import Path
from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

class ArchiveScriptCallback(TrainerCallback):
    def __init__(self, output_dir: str, config_file=None) -> None:
        super().__init__()
        self.output_dir = Path(output_dir)
        self.config_file = config_file
        
    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.is_world_process_zero:
            if self.config_file:
                shutil.copyfile(self.config_file, self.output_dir/os.path.basename(self.config_file))
            
            # Save the src directory
            os.makedirs(self.output_dir/"scripts", exist_ok=True)
            shutil.copytree("src/", self.output_dir/"scripts"/"src", dirs_exist_ok=True)
        return super().on_init_end(args, state, control, **kwargs)
        