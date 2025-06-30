import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
from typing import Tuple

class Expert(nn.Module):
    """
    An Expert is just a simple MLP (Multi-Layer Perceptron)
    Think of it as a specialist that learns to handle certain types of inputs
    """
    hidden_dim : int
    output_dim : int
    
    @nn.compact
    def __call__(self, x):
        h = nn.Dense(self.hidden_dim, name="w1")(x)
        h = nn.silu(h) # silu(x) = x * sigmoid(x)
        output = nn.Dense(self.output_dim, name="w2")(h)
        return output    
    
class SimpleGate(nn.Module):
    """
    Gate decides which expert to use. Usually it is decided per token
    But in this simplistic example, we will use a single gate for the entire sequence
    """
    num_experts : int

    @nn.compact
    def __call__(self, x):
        batch_size, seq_len, dim = x.shape
        
        # Average the input across the sequence dimension so that we make a decision per batch item
        # However in practice, we would use a per-token gate
        routing_input = jnp.mean(x, axis = 1) # [batch_size, dim]

        # Linear layer to get expert scores
        # This usually is trained to predict the expert weights accurately
        logits = nn.Dense(self.num_experts, name = "gate")(routing_input) # [batch_size, num_experts]

        # Softmax to normalise into probabilities
        expert_weights = jax.nn.softmax(logits, axis = -1) # [batch_size, num_experts]

        return expert_weights

class SimpleMoE(nn.Module):
    num_experts : int = 2
    hidden_dim : int = 512
    dim : int = 128

    @nn.compact
    def __call__(self, x):
        batch_size, seq_len, dim = x.shape
        print(f"MOE input shape: {x.shape}")

        # Get expert weights
        expert_weights = SimpleGate(num_experts = self.num_experts)(x) # [batch_size, num_experts]
        print(f"Expert weights shape: {expert_weights.shape}")

        # Create and apply all experts
        expert_outputs = []
        for i in range(self.num_experts):
            expert = Expert(hidden_dim = self.hidden_dim, output_dim = self.dim)
            output = expert(x) # [batch_size, seq_len, dim]
            expert_outputs.append(output)
            print(f"Expert {i} output shape: {output.shape}")
        
        # Combine expert outputs
        expert_outputs = jnp.stack(expert_outputs, axis = -1) # [batch_size, seq_len, dim, num_experts]

        # Broadcast expert weights to match the output shape
        # expert_weuights = [batch_size , num_experts]
        # expert_outputs = [batch_size, seq_len, dim, num_experts]
        expert_weights_broadcast = expert_weights[:, None, None, :] # [batch_size, 1, 1, num_experts]

        # Weighted combination of expert outputs
        combined_output = jnp.sum(expert_outputs * expert_weights_broadcast, axis = -1) # [batch_size, seq_len, dim]

        return combined_output, expert_weights