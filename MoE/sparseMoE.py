import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
from typing import Tuple

class Expert(nn.Module):
    hidden_dim : int
    dim : int 

    @nn.compact
    def __call__(self, x):
        h = nn.Dense(self.hidden_dim, name = "w1")(x)
        h = nn.silu(h)
        out = nn.Dense(self.dim, name = "w2")(h)

        return out
    
class TopKGate(nn.Module):
    """
    Selects only the top k experts
    Still doing it at a sequence level
    """
    num_experts : int
    top_k : int

    @nn.compact
    def __call__(self, x):
        batch_size, seq_len, dim = x.shape

        # Averaging across the sequence dimension
        routing_input = jnp.mean(x, axis = 1) # [batch_size, dim]

        # Adding a hidden layer for better gating decisions as opposed to a single linear layer
        gate_hidden = nn.Dense(dim, name = "gate_hidden")(routing_input) # [batch_size, dim]
        gate_hidden = nn.silu(gate_hidden)

        # Output scores for all experts
        logits = nn.Dense(self.num_experts, name = "gate_output")(gate_hidden) # [batch_size, num_experts]

        # Top-k selection
        top_k_logits, top_k_indices = jax.lax.top_k(logits, k = self.top_k) # [batch_size, top_k]

        # Getting softmax for only the selected experts
        top_k_weights = jax.nn.softmax(top_k_logits, axis = -1) # [batch_size, top_k]

        return top_k_weights, top_k_indices        

class SparseMoE(nn.Module):
    """
    Sparse MoE : only compute the selected top-k experts
    """
    dim : int = 128
    hidden_dim : int = 512
    num_experts : int = 4
    top_k : int = 2

    @nn.compact
    def __call__(self, x):
        batch_size, seq_len, dim = x.shape

        # Get top-k routing decisions
        top_k_weights, top_k_indices = TopKGate(
            num_experts = self.num_experts, top_k = self.top_k
            )(x) # [batch_size, top_k]
        
        # Creating all the experts
        experts = [Expert(hidden_dim = self.hidden_dim, dim = self.dim) for _ in range(self.num_experts)]

        # Initialise the output tensor
        output = jnp.zeros_like(x) # [batch_size, seq_len, dim]

        # Sparse computation
        for batch_idx in range(batch_size):
            print(f"Processing batch {batch_idx}")
            for k_idx in range(self.top_k):
                # Get which expert to use and its weight
                expert_idx = top_k_indices[batch_idx, k_idx]
                weight = top_k_weights[batch_idx, k_idx]

                print(f"Using expert {expert_idx} with weight {weight}")

                # Only compute this expert for this batch item
                single_batch_input = x[batch_idx:batch_idx+1] # [1, seq_len, dim] 
                # If we did x[batch_idx] then it would be [seq_len, dim] but expert expects [1, seq_len, dim]
                expert_output = experts[expert_idx](single_batch_input) # [1, seq_len, dim]

                # Add weight contribution to the output
                output = output.at[batch_idx].add(weight * expert_output.squeeze(0))
                # output[batch_idx] is [seq_len, dim] whereas expert_output is [1, seq_len, dim]
                # So we need to squeeze the expert_output to get [seq_len, dim]
    
        return output, (top_k_weights, top_k_indices)

# testing 
# def test_sparse_moe():
#     """Test the complete sparse MoE system"""
#     print("=== Testing Sparse MoE ===")
#     key = random.PRNGKey(0)
#     batch_size, seq_len, dim = 2, 3, 128
    
#     model = SparseMoE(dim=dim, hidden_dim=256, num_experts=4, top_k=2)
#     x = random.normal(key, (batch_size, seq_len, dim))
    
#     params = model.init(key, x)
#     output, (weights, indices) = model.apply(params, x)
    
#     print(f"\n=== Final Results ===")
#     print(f"Input shape: {x.shape}")
#     print(f"Output shape: {output.shape}")
    
#     print(f"\nRouting decisions:")
#     for i in range(batch_size):
#         selected_experts = indices[i]
#         expert_weights = weights[i]
#         print(f"Batch {i}: Experts {selected_experts} with weights {expert_weights}")
    
#     # Calculate efficiency gain
#     total_experts = model.num_experts
#     used_experts = model.top_k
#     efficiency_gain = total_experts / used_experts
#     print(f"\nEfficiency: Using {used_experts}/{total_experts} experts = {efficiency_gain:.1f}x speedup!")
    
#     assert output.shape == x.shape
#     print("✅ Sparse MoE test passed!")

# def compare_efficiency():
#     """Compare Simple MoE vs Sparse MoE efficiency"""
#     print("\n=== Efficiency Comparison ===")
    
#     configs = [
#         {"experts": 4, "top_k": 2},
#         {"experts": 8, "top_k": 2}, 
#         {"experts": 16, "top_k": 4},
#         {"experts": 64, "top_k": 8},
#     ]
    
#     for config in configs:
#         total = config["experts"]
#         used = config["top_k"]
#         speedup = total / used
#         print(f"MoE with {total} experts, top-{used}: {speedup:.1f}x speedup")

# test_sparse_moe()
# compare_efficiency()