from .kd_trainer import Seq2SeqKDTrainer
from .kd_arguments import Seq2SeqKDArguments
from .kd_trainer import KDLoggingCallback

KD_TRAINERS_DICT = dict(
    kd = Seq2SeqKDTrainer
)

KD_ARGS_DICT = dict(
    kd = Seq2SeqKDArguments
)