
import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig


class A2ModelConfig(PretrainedConfig):
    """Configuration object that stores hyperparameters that define the Transformer language model."""
    def __init__(self, vocab_size=None, hidden_size=None, intermediate_size=None, num_attention_heads=None, 
                 num_hidden_layers=None,
                 rope_theta=None, hidden_act='silu', max_position_embeddings=None, rms_norm_eps=None, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.num_attention_heads = num_attention_heads
        self.rope_theta = rope_theta
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers



class A2MLP(nn.Module):
    """The MLP layer of the Transformer. Uses the SwiGLU architecture."""
    def __init__(self, config):
        super().__init__()
        assert(config.hidden_act == 'silu')
        H = config.hidden_size
        I = config.intermediate_size
        # All linears bias=False for OLMo-2 compatibility.
        self.gate_proj = nn.Linear(H, I, bias=False)
        self.up_proj = nn.Linear(H, I, bias=False)
        self.down_proj = nn.Linear(I, H, bias=False)
        self.act_fn = nn.SiLU()
 

    def forward(self, hidden_states):
        # Element-wise product of two parallel projections, then project back.
        gated = self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        return self.down_proj(gated)

# This is optional, since you can use PyTorch's RMSNorm.
class A2RMSNorm(nn.Module):
    """RMS layer normalization."""
    def __init__(self, config):
        super().__init__()
        # TODO: Use config.rms_norm_eps
        # TODO: initalize weights here
        self.norm = nn.RMSNorm(
            normalized_shape=config.hidden_size,
            eps=config.rms_norm_eps,
            elementwise_affine=True,
        )

    def forward(self, hidden_states):
        return self.norm(hidden_states)


class A2Attention(nn.Module):
    """The multi-head attention layer of the Transformer. Uses standard scaled dot-product attention with causal masking."""
    
    def __init__(self, config):
        super().__init__()
        # TODO: set up W_q, W_k, W_v, W_o here
        # TODO: set up normalizers here
        assert config.hidden_size % config.num_attention_heads == 0, \
            'hidden_size must be divisible by num_attention_heads'
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
 
        # W_Q, W_K, W_V, W_O — all square, all bias=False.
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
 
        # OLMo-2 adds RMSNorm after Q and K projections (different from
        # the original Transformer; this is the "QK-norm" trick).
        self.q_norm = A2RMSNorm(config)
        self.k_norm = A2RMSNorm(config)

    def forward(self, hidden_states, rope_rotations):
        b, m, d = hidden_states.shape
        n_h, d_h = self.num_heads, self.head_dim
 
        # 1. Project to Q, K, V; apply QK-norm.
        q = self.q_norm(self.q_proj(hidden_states))   # (b, m, d)
        k = self.k_norm(self.k_proj(hidden_states))   # (b, m, d)
        v = self.v_proj(hidden_states)                # (b, m, d)
 
        # 2. Reshape into per-head form: (b, m, d) -> (b, n_h, m, d_h)
        q = q.view(b, m, n_h, d_h).transpose(1, 2)
        k = k.view(b, m, n_h, d_h).transpose(1, 2)
        v = v.view(b, m, n_h, d_h).transpose(1, 2)
 
        # 3. Apply RoPE rotations to q and k.
        q, k = apply_rotary_pos_emb(q, k, rope_rotations)
 
        # 4. Scaled dot-product attention with causal mask. PyTorch's
        # built-in handles scaling, masking, and softmax in one call.
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # attn_out: (b, n_h, m, d_h)
 
        # 5. Merge heads back: (b, n_h, m, d_h) -> (b, m, d)
        attn_out = attn_out.transpose(1, 2).reshape(b, m, d)
 
        # 6. Output projection.
        return self.o_proj(attn_out)


class A2DecoderLayer(nn.Module):
    """A complete Transformer decoder layer."""
    def __init__(self, config):
        super().__init__()
        # TODO: set up attention, MLP, and normalizers here.
        self.self_attn = A2Attention(config)
        self.mlp = A2MLP(config)
        self.post_attention_layernorm = A2RMSNorm(config)
        self.post_feedforward_layernorm = A2RMSNorm(config)

    def forward(self, hidden_states, rope_rotations):
        # Attention sublayer with residual.
        residual = hidden_states
        h = self.self_attn(hidden_states, rope_rotations)
        h = self.post_attention_layernorm(h)
        hidden_states = residual + h
 
        # MLP sublayer with residual.
        residual = hidden_states
        h = self.mlp(hidden_states)
        h = self.post_feedforward_layernorm(h)
        hidden_states = residual + h
 
        return hidden_states


class A2Transformer(PreTrainedModel):
    """A language model based on the Transformer architecture."""
    
    config_class = A2ModelConfig

    def __init__(self, config):
        super().__init__(config)

        self.rotary_emb = A2RotaryEmbedding(config)
        # TODO: Set up the other components here.
        # TODO: put all transformer decoder layers in a ModuleList.
        # Token embedding (vocab_size x hidden_size).
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
 
        # Stack of decoder layers, registered via ModuleList so their
        # parameters are picked up by .parameters().
        self.layers = nn.ModuleList([
            A2DecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
 
        # Final RMSNorm before the unembedding.
        self.norm = A2RMSNorm(config)
 
        # Unembedding (no bias, OLMo-2 compatible).
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
 
        # Loss function (same convention as A1: -100 means ignore).
        self.loss_func = nn.CrossEntropyLoss(ignore_index=-100)

        # This line should be called after you have set up all components.
        self.post_init()


    def forward(self, input_ids, labels=None):
        rope_rotations = self.rotary_emb(input_ids) # pass this to all the transformer decoder layers

        # TODO: Call embedding, transformer decoder layers, last normalizer, and unembedding.
        # TODO: Compute the loss as in Assignment 1 if labels is not None.
        # 2. Token embedding.
        hidden_states = self.embed_tokens(input_ids)           # (B, N, H)
 
        # 3. Run through every decoder layer.
        for layer in self.layers:
            hidden_states = layer(hidden_states, rope_rotations)
 
        # 4. Final norm + unembedding.
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)                    # (B, N, V)
 
        # 5. Loss with shift-by-one (same as A1).
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = self.loss_func(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
 
        from transformers.modeling_outputs import CausalLMOutput
        return CausalLMOutput(logits=logits, loss=loss)

#### RoPE implementation (copied and simplified from HuggingFace). ####

def apply_rotary_pos_emb(q, k, rope_rotations, unsqueeze_dim=1):
    """Applies precomputed RoPE rotations to the query and key representations."""
    assert(q.shape == k.shape)
    assert(len(q.shape) == 4)
    cos, sin = rope_rotations
    assert(q.shape[2] == cos.shape[1])
    assert(q.shape[3] == cos.shape[2])    
    q_type, k_type = q.dtype, k.dtype
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(q_type), k_embed.to(k_type)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

class A2RotaryEmbedding(nn.Module):
    """RoPE position representation for use in Transformer attention."""

    def __init__(self, config, device=None):
        super().__init__()
        rope_theta = config.rope_theta
        head_dim = config.hidden_size // config.num_attention_heads
        partial_rotary_factor = 1.0
        dim = int(head_dim * partial_rotary_factor)
        self.inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))

    @torch.no_grad()
    def forward(self, x):
        position_ids = torch.arange(0, x.shape[1], device=x.device).unsqueeze(0)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
            return cos, sin
