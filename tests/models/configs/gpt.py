from dataclasses import dataclass
from typing import Dict

@dataclass
class GPTConfig:
    n_embd: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_positions: int = 32
    vocab_size: int = 1000
    resid_pdrop: float = 0.0
    embd_pdrop: float = 0.0
    attn_pdrop: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "n_embd": self.n_embd,
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "n_positions": self.n_positions,
            "vocab_size": self.vocab_size,
            "resid_pdrop": self.resid_pdrop,
            "embd_pdrop": self.embd_pdrop,
            "attn_pdrop": self.attn_pdrop
        }
