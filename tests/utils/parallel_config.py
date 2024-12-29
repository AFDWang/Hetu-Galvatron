from dataclasses import dataclass
from typing import List
import json

@dataclass
class ParallelConfig:
    pp_deg: int
    tp_sizes_enc: List[int]
    tp_consecutive_flags: List[int]
    dp_types_enc: List[str]
    use_sp: List[int]
    checkpoint: List[int]
    global_bsz: int
    chunks: int
    pp_division: List[int]
    pipeline_type: str
    default_dp_type: str
    vtp: int
    vsp: int

    def to_dict(self):
        return {
            "pp_deg": self.pp_deg,
            "tp_sizes_enc": ",".join(map(str, self.tp_sizes_enc)),
            "tp_consecutive_flags": ",".join(map(str, self.tp_consecutive_flags)),
            "dp_types_enc": ",".join(map(str, self.dp_types_enc)),
            "use_sp": ",".join(map(str, self.use_sp)),
            "checkpoint": ",".join(map(str, self.checkpoint)),
            "global_bsz": self.global_bsz,
            "chunks": self.chunks,
            "pp_division": ",".join(map(str, self.pp_division)),
            "pipeline_type": self.pipeline_type,
            "default_dp_type": self.default_dp_type,
            "vtp": self.vtp,
            "vsp": self.vsp
        }