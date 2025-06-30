import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
import optax
from flax.training import train_state

def create_causal_mask(seq_len):
    """
    Create causal mask to prevent attending to future positions.
    Used in autoregressive models like GPT.
    
    Returns:
        mask: [seq_len, seq_len] with -inf in upper triangle
    """
    # Upper triangular matrix with -inf (can't attend to future)
    mask = jnp.triu(jnp.full((seq_len, seq_len), -jnp.inf), k=1)
    return mask

class MultiHeadLatentAttention(nn.Module):
    d_model: int = 128
    num_heads: int = 8
    
    # Latent space dimensions
    kv_lora_rank: int = 64      # Shared key-value latent dimension
    q_lora_rank: int = 32       # Query low-rank adaptation dimension
    
    # Attention head dimensions
    qk_nope_head_dim: int = 32  # Query-key dimension without positional encoding
    qk_rope_head_dim: int = 16  # Query-key dimension with rotary positional encoding
    v_head_dim: int = 32        # Value head dimension

    def apply_rope_encoding(self, x, start_pos = 0):
        """
        Apply Rotary Positional Encoding (RoPE)
        This is a simplified version - full RoPE uses complex exponentials
        """
        seq_len , head_dim = x.shape[1], x.shape[-1]

        # Create position indices
        positions = jnp.arange(start_pos, start_pos + seq_len)

        # Create frequency basis (simplified)
        freqs = 1.0 / (10000 ** (jnp.arange(0, head_dim, 2) / head_dim))

        # Compute position encodings
        pos_enc = positions[:, None] * freqs[None, :]  # [seq_len, head_dim//2]
        
        # Apply rotary transformation (simplified - real RoPE uses complex rotations)
        sin_enc = jnp.sin(pos_enc)
        cos_enc = jnp.cos(pos_enc)
        
        # Broadcast to match input shape
        sin_enc = jnp.broadcast_to(sin_enc[None, :, None, :], x.shape[:-1] + (head_dim//2,))
        cos_enc = jnp.broadcast_to(cos_enc[None, :, None, :], x.shape[:-1] + (head_dim//2,))
        
        # Apply rotation (simplified version)
        x_even = x[..., ::2] * cos_enc - x[..., 1::2] * sin_enc
        x_odd = x[..., ::2] * sin_enc + x[..., 1::2] * cos_enc
        
        # Interleave even and odd components
        result = jnp.stack([x_even, x_odd], axis=-1).reshape(x.shape)
        
        return result

    @nn.compact
    def __call__(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        assert self.qk_rope_head_dim % 2 == 0, f"qk_rope_head_dim must be even, got {self.qk_rope_head_dim}"
        assert self.num_heads * self.v_head_dim == self.d_model, \
            f"num_heads * v_head_dim ({self.num_heads * self.v_head_dim}) must equal d_model ({self.d_model})"

        # Downsample KV to latent space
        # Project input into shared latent space + positional component
        kv_latent_and_rope = nn.Dense(
            self.kv_lora_rank + self.qk_rope_head_dim,
            name = 'wkv_a'
        )(x) # [batch, seq_len, kv_lora_rank + qk_rope_head_dim]

        # Split into content and positional components
        kv_latent, k_rope_input = jnp.split(
            kv_latent_and_rope,
            # [self.kv_lora_rank, self.qk_rope_head_dim],
            [self.kv_lora_rank],
            axis = -1
        ) # k_rope_input: [batch, seq_len, qk_rope_head_dim]

        # Layer normalisation on latent space
        kv_latent_norm = nn.LayerNorm(name = 'kv_norm')(kv_latent)

        # Upsample latent to all heads' K_nope and V
        # Project shared latent to all heads' K_nope and V
        kv_projected = nn.Dense(
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            name = 'wkv_b'
        )(kv_latent_norm)

        # Reshape for multi-head structure
        kv_projected = kv_projected.reshape(
            batch_size,
            seq_len,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim
        ) # [batch, seq_len, num_heads, qk_nope_head_dim + v_head_dim]
        
        # Split into content keys and values
        k_nope, v = jnp.split(
            kv_projected,
            # [self.qk_nope_head_dim, self.v_head_dim],
            [self.qk_nope_head_dim],
            axis = -1
        ) # k_nope : [batch, seq_len, num_heads, qk_nope_head_dim]

        # Query projection with optional LoRA decomposition
        if self.q_lora_rank == 0:
            # Direct projection
            q_projected = nn.Dense(
                self.num_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim),
                name = 'wq'
            )(x)
        else:
            # LoRA style decomposition
            q_down = nn.Dense(self.q_lora_rank, name = 'wq_a')(x) # Down projection
            q_down_norm = nn.LayerNorm(name = 'q_norm')(q_down)
            q_projected = nn.Dense(
                self.num_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim),
                name = 'wq_b'
            )(q_down_norm)
          
        q_projected = q_projected.reshape(
            batch_size,
            seq_len,
            self.num_heads,
            self.qk_nope_head_dim + self.qk_rope_head_dim
        )
        
        q_nope, q_rope_input = jnp.split(
            q_projected,
            # [self.qk_nope_head_dim, self.qk_rope_head_dim],
            [self.qk_nope_head_dim],
            axis = -1
        ) # q_nope : [batch, seq_len, num_heads, qk_nope_head_dim]

        # Applying rope to key positional components
        # k_rope_input is shared across heads so we expand it to match q_rope_input which has separate components for each head
        k_rope_input_expanded = jnp.expand_dims(k_rope_input, axis = 2) # [batch, seq_len, qk_rope_head_dim] -> [batch, seq_len, 1, qk_rope_head_dim]

        # Replicate across all heads ([batch, seq_len, 1, qk_rope_head_dim] -> [batch, seq_len, num_heads, qk_rope_head_dim])
        target_shape = (batch_size, seq_len, self.num_heads, self.qk_rope_head_dim)
        k_rope_input_expanded = jnp.broadcast_to(k_rope_input_expanded, target_shape)
        
        # Apply RoPe to query and key positional components
        q_rope = self.apply_rope_encoding(q_rope_input)
        k_rope = self.apply_rope_encoding(k_rope_input_expanded)
        
        # Content based attention scores(semantic similarity)
        content_scores = jnp.einsum('bshd,bthd->bsht', q_nope, k_nope)
        content_scale = 1.0 / jnp.sqrt(self.qk_nope_head_dim)
        content_scores = content_scores * content_scale

        # Position based attention scores(positional similarity)
        position_scores = jnp.einsum('bshd,bthd->bsht', q_rope, k_rope) 
        position_scale = 1.0 / jnp.sqrt(self.qk_rope_head_dim)
        position_scores = position_scores * position_scale

        # Combining content and positional attention
        attention_scores = content_scores + position_scores

        # Apply causal mask for autoregressive generation
        if mask is not None:
            attention_scores = attention_scores + mask[None, :, None, :]
        
        # Softmax normalization
        attention_weights = jax.nn.softmax(attention_scores, axis = -1)

        # Apply attention to values
        attended_values = jnp.einsum('bsht,bthd->bshd', attention_weights, v)

        # Reshaping attended values for output projection
        # We are essentially concatenating all heads' attended values into a single tensor
        attended_flat = attended_values.reshape(batch_size, seq_len, -1)

        # Final linear projection back to model dimension
        output = nn.Dense(self.d_model, name = 'wo')(attended_flat)
        
        return output, attention_weights
    
    