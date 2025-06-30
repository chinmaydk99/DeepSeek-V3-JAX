# Multi-Head Latent Attention (MLA) in JAX/Flax

This document provides a detailed explanation of the `MultiHeadLatentAttention` module implemented in JAX/Flax. This module is inspired by the attention mechanism in recent advanced language models like DeepSeek-V2. It introduces an efficient and powerful variant of multi-head attention by decoupling content-based and position-based attention computations and utilizing a shared latent space for keys and values.

---

## 🔑 Core Concepts

The architecture is built on three main innovative ideas:

### 1. Shared Key-Value (KV) Bottleneck 🎯
Instead of projecting the input sequence independently for each head's keys and values, this model first projects the input into a shared, low-dimensional latent space (a bottleneck). This latent representation is then used to generate the keys and values for all attention heads. 

**Benefits:**
- Significantly reduces parameters (especially for models with many heads)
- Lower computational cost
- Better parameter sharing across heads

### 2. Decoupling Content and Positional Attention 📍
The model separates attention scores into two distinct components:

- **Content Attention**: Captures semantic similarity between tokens, independent of their positions
  - Uses query and key vectors without positional encoding (`q_nope`, `k_nope`)
- **Positional Attention**: Captures relative positional information between tokens  
  - Uses separate query and key vectors enriched with RoPE (`q_rope`, `k_rope`)

The final attention score = content attention + positional attention, allowing independent weighting of semantic and positional similarity.

### 3. Low-Rank Adaptation (LoRA) for Queries 🔧
Query projection can optionally use LoRA-style decomposition:
- Input → Down-projection to low-rank space (`q_lora_rank`)
- Low-rank space → Up-projection to full query dimension
- Enables parameter-efficient adaptation and reduces model size

---

## 📐 Mathematical Formulation

### Dimension Definitions

| Symbol | Description | Code Parameter |
|--------|-------------|----------------|
| $B$ | Batch size | - |
| $S$ | Sequence length | - |
| $D$ | Model dimension | `d_model` |
| $N_h$ | Number of heads | `num_heads` |
| $R_{kv}$ | KV latent rank | `kv_lora_rank` |
| $R_q$ | Query LoRA rank | `q_lora_rank` |
| $H_{nope}$ | Content attention head dim | `qk_nope_head_dim` |
| $H_{rope}$ | Positional attention head dim | `qk_rope_head_dim` |
| $H_v$ | Value head dimension | `v_head_dim` |

**Input**: $X \in \mathbb{R}^{B \times S \times D}$

---

### 1. Key-Value (KV) Path 🗝️

**Step 1: Down-projection to shared latent space**
$$X_{kv\_combined} = X W_{kv\_a}$$
where $W_{kv\_a} \in \mathbb{R}^{D \times (R_{kv} + H_{rope})}$

**Step 2: Split into content and positional components**
$$KV_{latent}, K_{rope\_input} = \text{split}(X_{kv\_combined})$$
- $KV_{latent} \in \mathbb{R}^{B \times S \times R_{kv}}$
- $K_{rope\_input} \in \mathbb{R}^{B \times S \times H_{rope}}$

**Step 3: Normalize and up-project to all heads**
$$KV_{latent\_norm} = \text{LayerNorm}(KV_{latent})$$
$$KV_{proj} = KV_{latent\_norm} W_{kv\_b}$$
where $W_{kv\_b} \in \mathbb{R}^{R_{kv} \times N_h(H_{nope} + H_v)}$

**Step 4: Reshape and split to get final tensors**
- $K_{nope} \in \mathbb{R}^{B \times S \times N_h \times H_{nope}}$ (content keys)
- $V \in \mathbb{R}^{B \times S \times N_h \times H_v}$ (values)

---

### 2. Query (Q) Path with LoRA 🎯

**Step 1: Down-projection (LoRA)**
$$Q_{down} = X W_{q\_a}$$
where $W_{q\_a} \in \mathbb{R}^{D \times R_q}$

**Step 2: Normalize and up-project**
$$Q_{down\_norm} = \text{LayerNorm}(Q_{down})$$
$$Q_{proj} = Q_{down\_norm} W_{q\_b}$$
where $W_{q\_b} \in \mathbb{R}^{R_q \times N_h(H_{nope} + H_{rope})}$

**Step 3: Reshape and split**
- $Q_{nope} \in \mathbb{R}^{B \times S \times N_h \times H_{nope}}$ (content queries)
- $Q_{rope\_input} \in \mathbb{R}^{B \times S \times N_h \times H_{rope}}$ (positional query input)

---

### 3. RoPE and Attention Score Calculation 🔄

**Step 1: Apply Rotary Positional Encoding**
$$Q_{rope} = \text{RoPE}(Q_{rope\_input})$$
$$K_{rope} = \text{RoPE}(\text{broadcast}(K_{rope\_input}))$$

Note: $K_{rope\_input}$ is shared across heads and broadcast to match query dimensions.

**Step 2: Compute attention scores**

*Content attention (semantic similarity):*
$$\text{Scores}_{content} = \frac{Q_{nope} K_{nope}^T}{\sqrt{H_{nope}}}$$

*Positional attention (positional similarity):*
$$\text{Scores}_{position} = \frac{Q_{rope} K_{rope}^T}{\sqrt{H_{rope}}}$$

*Combined attention scores:*
$$\text{AttentionScores} = \text{Scores}_{content} + \text{Scores}_{position}$$

---

### 4. Output Generation 📤

**Step 1: Apply mask and softmax**
$$\text{AttentionWeights} = \text{softmax}(\text{AttentionScores} + \text{Mask})$$

**Step 2: Apply attention to values**
$$\text{AttendedValues} = \text{AttentionWeights} \cdot V$$

**Step 3: Final output projection**
$$\text{Output} = \text{reshape}(\text{AttendedValues}) W_o$$
where $W_o \in \mathbb{R}^{N_h H_v \times D}$

---

## 💻 JAX/Flax Implementation Details

Let's walk through the `MultiHeadLatentAttention` module's `__call__` method step by step.

### Class Definition and Parameters
```python
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
```

### Implementation Walkthrough

#### 🗝️ 1. KV Path Implementation
```python
# Downsample KV to latent space
kv_latent_and_rope = nn.Dense(
    self.kv_lora_rank + self.qk_rope_head_dim, name='wkv_a')(x)

# Split into content and positional components
kv_latent, k_rope_input = jnp.split(kv_latent_and_rope, [self.kv_lora_rank], axis=-1)

# Layer normalisation on latent space
kv_latent_norm = nn.LayerNorm(name='kv_norm')(kv_latent)

# Upsample latent to all heads' K_nope and V
kv_projected = nn.Dense(
    self.num_heads * (self.qk_nope_head_dim + self.v_head_dim), name='wkv_b')(kv_latent_norm)

# Reshape and split
kv_projected = kv_projected.reshape(
    batch_size, seq_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
k_nope, v = jnp.split(kv_projected, [self.qk_nope_head_dim], axis=-1)
```

**Key Points:**
- `wkv_a` implements the down-projection to shared latent space
- LayerNorm is applied in the latent space for stability
- `wkv_b` up-projects to generate all heads' keys and values simultaneously

#### 🎯 2. Query Path Implementation
```python
if self.q_lora_rank == 0:
    # Direct projection
    q_projected = nn.Dense(
        self.num_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim), name='wq')(x)
else:
    # LoRA style decomposition
    q_down = nn.Dense(self.q_lora_rank, name='wq_a')(x)
    q_down_norm = nn.LayerNorm(name='q_norm')(q_down)
    q_projected = nn.Dense(
        self.num_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim), name='wq_b')(q_down_norm)
    
q_projected = q_projected.reshape(
    batch_size, seq_len, self.num_heads, self.qk_nope_head_dim + self.qk_rope_head_dim)
q_nope, q_rope_input = jnp.split(q_projected, [self.qk_nope_head_dim], axis=-1)
```

**Key Points:**
- Optional LoRA decomposition with `wq_a` (down) and `wq_b` (up)
- LayerNorm applied in the low-rank space when using LoRA
- Final tensors split into content and positional components

#### 🔄 3. RoPE and Attention Computation
```python
# Expand and broadcast k_rope_input to match query's head dimension
k_rope_input_expanded = jnp.expand_dims(k_rope_input, axis=2)
k_rope_input_expanded = jnp.broadcast_to(k_rope_input_expanded, target_shape)

# Apply RoPE
q_rope = self.apply_rope_encoding(q_rope_input)
k_rope = self.apply_rope_encoding(k_rope_input_expanded)

# Content scores
content_scores = jnp.einsum('bshd,bthd->bsht', q_nope, k_nope)
content_scores = content_scores * (1.0 / jnp.sqrt(self.qk_nope_head_dim))

# Position scores
position_scores = jnp.einsum('bshd,bthd->bsht', q_rope, k_rope)
position_scores = position_scores * (1.0 / jnp.sqrt(self.qk_rope_head_dim))

# Combine scores
attention_scores = content_scores + position_scores
```

**Key Points:**
- Shared `k_rope_input` is broadcast to all heads for efficiency
- `jnp.einsum` performs efficient batched matrix multiplication
- Separate scaling factors for content and positional attention
- Additive combination of content and positional scores

#### 📤 4. Final Output Generation
```python
# Apply causal mask
if mask is not None:
    attention_scores = attention_scores + mask[None, :, None, :]

# Softmax normalization
attention_weights = jax.nn.softmax(attention_scores, axis=-1)

# Apply attention to values
attended_values = jnp.einsum('bsht,bthd->bshd', attention_weights, v)

# Reshape and final projection
attended_flat = attended_values.reshape(batch_size, seq_len, -1)
output = nn.Dense(self.d_model, name='wo')(attended_flat)

return output, attention_weights
```

**Key Points:**
- Optional causal masking for autoregressive generation
- Standard softmax attention mechanism
- Final projection `wo` back to model dimension
- Returns both output and attention weights for analysis

---

## 🚀 Usage Example

Here's a complete example of how to use the `MultiHeadLatentAttention` module:

```python
import jax
import jax.numpy as jnp
from jax import random
from multihead_latent_attention import MultiHeadLatentAttention, create_causal_mask

# Model hyperparameters
d_model = 128
num_heads = 8
batch_size = 4
seq_len = 16

# Create dummy input data
key = random.PRNGKey(42)
x = random.normal(key, (batch_size, seq_len, d_model))

# Instantiate the attention module
mla = MultiHeadLatentAttention(
    d_model=d_model,
    num_heads=num_heads,
    kv_lora_rank=64,
    q_lora_rank=32,
    qk_nope_head_dim=32,
    qk_rope_head_dim=16,
    v_head_dim=d_model // num_heads  # Ensure compatibility
)

# Initialize parameters
params = mla.init(key, x)['params']

# Create causal mask for autoregressive attention
causal_mask = create_causal_mask(seq_len)

# Forward pass
output, attention_weights = mla.apply({'params': params}, x, mask=causal_mask)

# Verify shapes
print(f"Input shape:             {x.shape}")
print(f"Output shape:            {output.shape}")  
print(f"Attention weights shape: {attention_weights.shape}")

# Expected output:
# Input shape:             (4, 16, 128)
# Output shape:            (4, 16, 128)
# Attention weights shape: (4, 16, 8, 16)
```

---

## 🔧 Key Features & Benefits

| Feature | Benefit |
|---------|---------|
| **Shared KV Bottleneck** | Reduces parameters by ~40-60% compared to standard MHA |
| **Content/Position Separation** | Better interpretability and control over attention patterns |
| **LoRA Query Projection** | Further parameter reduction with minimal performance loss |
| **RoPE Integration** | Superior handling of long sequences and positional relationships |
| **JAX/Flax Implementation** | High performance, automatic differentiation, and easy scaling |

---

## 📚 References

This implementation is inspired by:
- **DeepSeek-V3**: Multi-Head Latent Attention architecture
- **RoPE**: Rotary Position Embedding for improved positional encoding
- **LoRA**: Low-Rank Adaptation for parameter-efficient fine-tuning

For more details on the theoretical foundations, refer to the original papers on these techniques.