import torch
from torch import nn
import torch.distributed as dist
from transformers import LlamaConfig

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead

class Llama3_1Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: dict | None = None,
    ) -> None:
        super().__init__()
        if dist.is_initialized():
            tp_size = dist.get_world_size()
        else:
            tp_size = 1
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        # self.rotary_emb = get_rope(
        #     self.head_dim,
        #     rotary_dim=self.head_dim,
        #     max_position=max_position,
        #     base=rope_theta,
        #     if rope_scaling is not None:
        #         rope_scaling_type=rope_scaling.get("rope_type"),
        #         rope_scaling_factor=rope_scaling.get("factor"),
        #         low_freq_factor=rope_scaling.get("low_freq_factor", 1.0),
        #         high_freq_factor=rope_scaling.get("high_freq_factor", 4.0),
        #         original_max_position_embeddings=rope_scaling.get("original_max_position_embeddings"),
        # )

        rope_kwargs = {}

        if rope_scaling is not None:
            rope_kwargs["rope_scaling_type"] = rope_scaling.get("rope_type")
            rope_kwargs["rope_scaling_factor"] = rope_scaling.get("factor")
            rope_kwargs["low_freq_factor"] = rope_scaling.get("low_freq_factor", 1.0)
            rope_kwargs["high_freq_factor"] = rope_scaling.get("high_freq_factor", 4.0)
            rope_kwargs["original_max_position_embeddings"] = rope_scaling.get("original_max_position_embeddings")
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            **rope_kwargs
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q_by_head = q.view(-1, self.num_heads, self.head_dim)
        # q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.view(-1, self.num_kv_heads, self.head_dim)
        # k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o)
        return output


class Llama3_1MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Llama3_1DecoderLayer(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
    ) -> None:
        super().__init__()
        self.self_attn = Llama3_1Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', False),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = Llama3_1MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Llama3_1Model(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Llama3_1DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        print("Llama3_1Model enters forward")
        hidden_states = self.embed_tokens(input_ids)
        print(f"Hidden states after embedding: {hidden_states}")
        residual = None
        for i, layer in enumerate(self.layers):
            hidden_states, residual = layer(positions, hidden_states, residual)

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


    # def forward(
    #     self,
    #     input_ids: torch.Tensor,
    #     positions: torch.Tensor,
    # ) -> torch.Tensor:

    #     hidden_states = self.embed_tokens(input_ids)
    #     residual = None

    #     # --- DEBUG PRINTS GO HERE ---
    #     if dist.is_initialized():
    #         if dist.get_rank() == 0:
    #             print(f"--- Entering Transformer Layers Loop ---", flush=True)
    #         dist.barrier()

    #     for i, layer in enumerate(self.layers):
    #         if dist.is_initialized():
    #             if dist.get_rank() == 0:
    #                 print(f"--> Starting Layer {i}", flush=True)
    #             dist.barrier()

    #         hidden_states, residual = layer(positions, hidden_states, residual)

    #         if dist.is_initialized():
    #             dist.barrier() # Sync after each layer to isolate a hang
    #             if dist.get_rank() == 0:
    #                 print(f"<-- Finished Layer {i}", flush=True)
        
    #     if dist.is_initialized():
    #         if dist.get_rank() == 0:
    #             print(f"--- Exited Transformer Layers Loop ---", flush=True)
    #         dist.barrier()
    #     # --- END DEBUG PRINTS ---

    #     hidden_states, _ = self.norm(hidden_states, residual)
    #     return hidden_states


class LlamaForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: LlamaConfig
    ) -> None:
        super().__init__()
        self.model = Llama3_1Model(config)
        self.config = config
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        print("Llama enters forward")
        print(f"Input IDs: {input_ids}, Positions: {positions}")
        hidden_states = self.model(input_ids, positions)
        # print("Llama ends forward")
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        is_verify: bool = False,
    ) -> torch.Tensor:
        if self.config.tp_size > 1:
            dist.broadcast(hidden_states, src=0, group=self.config.tp_group)

        logits = self.lm_head(hidden_states, is_verify)
        return logits
