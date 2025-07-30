from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)
    x1, x2 = torch.chunk(x.to(torch.float32), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        rope_scaling: dict | None = None,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size

        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        if rope_scaling is not None:
            self.rope_scaling = rope_scaling
            # 설정값 가져오기
            scaling_type = rope_scaling.get("rope_type")
            scaling_factor = rope_scaling.get("factor")
            low_freq_factor = rope_scaling.get("low_freq_factor", 1.0)
            high_freq_factor = rope_scaling.get("high_freq_factor", 4.0)
            original_max_pos = rope_scaling.get("original_max_position_embeddings")

            # 컨텍스트 확장 비율 계산
            scale = max_position_embeddings / original_max_pos

            if scale > 1.0 and scaling_type == "llama3":
                lam = (high_freq_factor - low_freq_factor) / (base - 1)
                inv_freq_ext = inv_freq / (low_freq_factor + lam * (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim) - 1))
                
                # 보간 계수(correction factor) 계산
                mscale = (scale / scaling_factor)**(rotary_dim / (rotary_dim - 2))
                
                # 최종 역주파수 계산
                inv_freq = inv_freq_ext * mscale

        # 위치 텐서 생성
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = positions.size(0)
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)
        query = apply_rotary_emb(query, cos, sin).view(query_shape)
        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_size)
        key = apply_rotary_emb(key, cos, sin).view(key_shape)
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    # rope_scaling: dict | None = None,
    rope_scaling_type: str | None = None,
    rope_scaling_factor: float | None = None,
    low_freq_factor: float | None = None,
    high_freq_factor: float | None = None,
    original_max_position_embeddings: int | None = None,
):
    rope_scaling = None
    if rope_scaling_type:
        rope_scaling = {
            "rope_type": rope_scaling_type,
            "factor": rope_scaling_factor,
            "low_freq_factor": low_freq_factor,
            "high_freq_factor": high_freq_factor,
            "original_max_position_embeddings": original_max_position_embeddings,
        }

    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base, rope_scaling)
    return rotary_emb
