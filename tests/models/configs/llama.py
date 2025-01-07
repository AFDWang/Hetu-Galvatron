from dataclasses import dataclass
from typing import Dict

@dataclass
class LlamaConfig:
    dim: int = 128
    multiple_of: int = 4
    n_heads: int = 4
    n_layers: int = 4
    norm_eps: float = 1e-5
    vocab_size: int = 1000
    n_positions: int = 32

    def to_dict(self) -> Dict:
        return {
            "dim": self.dim,
            "multiple_of": self.multiple_of,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "norm_eps": self.norm_eps,
            "vocab_size": self.vocab_size,
            "n_positions": self.n_positions
        }

@dataclass
class Llama2Config:
    dim: int = 128
    multiple_of: int = 4
    n_heads: int = 4
    n_kv_heads: int = 2
    n_layers: int = 4
    norm_eps: float = 1e-5
    vocab_size: int = 1000
    n_positions: int = 32

    def to_dict(self) -> Dict:
        return {
            "dim": self.dim,
            "multiple_of": self.multiple_of,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "n_layers": self.n_layers,
            "norm_eps": self.norm_eps,
            "vocab_size": self.vocab_size,
            "n_positions": self.n_positions
        }

@dataclass
class LlamaConfigSearch:
    dim: int = 3584
    multiple_of: int = 256
    n_heads: int = 28
    n_layers: int = 28
    norm_eps: float = 1e-06
    vocab_size: int = 152064
    n_positions: int = 4096

    def to_dict(self) -> Dict:
        return {
            "dim": self.dim,
            "multiple_of": self.multiple_of,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "norm_eps": self.norm_eps,
            "vocab_size": self.vocab_size,
            "n_positions": self.n_positions
        }
