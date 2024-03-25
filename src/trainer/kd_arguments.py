from dataclasses import dataclass, field

@dataclass
class Seq2SeqKDArguments:
    reverse_kld: bool = field(default=False, metadata={"help": "Whether to use reverse KL divergence."}) 
    kd_ratio: float = field(default=0.5, metadata={"help": "Weight for KD loss."}) 
    kd_temperature: float = field(default=1.0, metadata={"help": "Teamperature for computing KL divergence."})