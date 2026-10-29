"""gpt-scratch: a from-scratch decoder-only GPT in pure PyTorch."""

from .config import ModelConfig, TrainConfig
from .model import GPT
from .tokenizer import ByteBPETokenizer, train_bpe

__all__ = ["GPT", "ModelConfig", "TrainConfig", "ByteBPETokenizer", "train_bpe"]
__version__ = "0.1.0"
