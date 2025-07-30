import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaConfig
import torch.distributed as dist

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import ParallelLMHead


class EagleLlama3_1Attention(nn.Module):

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
        rope_scaling: tuple | None = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim or hidden_size // num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        
        qkv_output_dim = self.q_size + 2 * self.kv_size
        self.qkv_proj = nn.Linear(hidden_size, qkv_output_dim, bias=qkv_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=False)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: tuple | None = None
    ) -> tuple[torch.Tensor, tuple]:
        batch_size, seq_len, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        positions = positions.view(-1)
        q = q.view(-1, self.q_size)
        k = k.view(-1, self.kv_size)

        q, k = self.rotary_emb(positions, q, k)
        
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)
        
        new_kv_cache = (k, v)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        if self.num_kv_heads != self.num_heads:
            num_reps = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(num_reps, dim=1)
            v = v.repeat_interleave(num_reps, dim=1)

        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        
        output = self.o_proj(attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))
        return output, new_kv_cache


class EagleLlama3_1MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        activated_hidden_states = F.silu(gate) * up
        x = self.down_proj(activated_hidden_states)
        return x


class EagleLlama3_1DecoderLayer(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
    ) -> None:
        super().__init__()
        self.self_attn = EagleLlama3_1Attention(
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
        self.mlp = EagleLlama3_1MLP(
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
        kv_cache: tuple | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple]:
        
        residual = hidden_states
        normalized_hidden_states = self.input_layernorm(hidden_states)
        attn_output, new_kv_cache = self.self_attn(positions, normalized_hidden_states, kv_cache)
        hidden_states = attn_output + residual

        residual = hidden_states
        normalized_hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(normalized_hidden_states)
        hidden_states = mlp_output + residual

        return hidden_states, None, new_kv_cache


class EagleLlama3_1Model(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
    ) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.fc = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.layers = nn.ModuleList([EagleLlama3_1DecoderLayer(config)])
        self.norm = nn.Identity()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
        kv_cache: tuple | None = None
    ) -> tuple[torch.Tensor, tuple]:
        # print("Entered EAGLE Model Forward-1")
        print(f"input id is {input_ids}\n positions is {positions} and hidden_states is {hidden_states}")
        inter_hidden_states = self.embed_tokens(input_ids)
        # print("Entered EAGLE Model Forward-2")

        if hidden_states is not None:
            hidden_states = hidden_states.to(inter_hidden_states.dtype)
            
            # *** THE FIX IS HERE ***
            # Ensure hidden_states is 3D before concatenation.
            # If it's 2D (from the main model), add a sequence dimension.
            if hidden_states.dim() == 2:
                hidden_states = hidden_states.unsqueeze(1)
            # Now hidden_states is guaranteed to be 3D [batch, 1, hidden_size]
            
            # Concatenate the 3D tensors.
            cat_input = torch.cat((inter_hidden_states, hidden_states), dim=-1)
        else:
            zero_hs = torch.zeros_like(inter_hidden_states)
            cat_input = torch.cat((inter_hidden_states, zero_hs), dim=-1)

        # print("Entered EAGLE Model Forward-3")
        hidden_states = self.fc(cat_input)
        
        hidden_states, _, new_kv_cache = self.layers[0](positions, hidden_states, None, kv_cache)
        
        hidden_states = self.norm(hidden_states)
        # print("Entered EAGLE Model Forward-4")
        return hidden_states, new_kv_cache


class EagleLlama3_1ForCausalLM(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.model = EagleLlama3_1Model(config)
        self.config = config
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
        kv_cache: tuple | None = None
    ) -> tuple[torch.Tensor, tuple]:
        # print("Entered CausalLM Forward")
        final_hidden_states, new_kv_cache = self.model(input_ids, positions, hidden_states, kv_cache)
        return final_hidden_states, new_kv_cache

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # print("Enter EAGLE compute_logits")
        logits = self.lm_head(hidden_states)
        # print("Exit EAGLE compute_logits")
        return logits

    @torch.inference_mode()
    def propose(
        self,
        last_tokens: list[int],
        initial_hidden_state: torch.Tensor,
        num_spec_tokens: int,
        start_positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # print("Entered Propose")
        device = next(self.model.parameters()).device
        
        input_ids = torch.tensor(last_tokens, dtype=torch.long, device=device).unsqueeze(1)
        current_hidden_state = initial_hidden_state
        positions = start_positions.clone()
        
        collected_tokens = []
        collected_logits = []
        draft_kv_cache = None

        for i in range(num_spec_tokens):
            # print(f"Proposing token {i+1}/{num_spec_tokens}")
            current_positions = (positions + i).unsqueeze(1)
            
            next_hidden_state, draft_kv_cache = self.forward(
                input_ids=input_ids,
                positions=current_positions,
                hidden_states=current_hidden_state,
                kv_cache=draft_kv_cache
            )
            
            logits = self.compute_logits(next_hidden_state)
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            
            collected_tokens.append(next_token)
            collected_logits.append(logits.squeeze(1))

            input_ids = next_token.unsqueeze(1)
            current_hidden_state = next_hidden_state
            # print(f"Proposed token:",next_token)

        candidate_tokens = torch.stack(collected_tokens, dim=1)
        draft_logits = torch.stack(collected_logits, dim=1)
        # print("The candidate tokens are:", candidate_tokens)
        return candidate_tokens, draft_logits
