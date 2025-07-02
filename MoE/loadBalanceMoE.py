import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
from typing import Tuple, Dict
import optax
from flax.training import train_state

class Expert(nn.Module):
    """Single expert - same as before"""  
    hidden_dim: int
    dim: int
    
    @nn.compact
    def __call__(self, x):
        h = nn.Dense(self.hidden_dim, name='w1')(x)
        h = nn.silu(h)
        return nn.Dense(self.dim, name='w2')(h)


def compute_load_balance_loss(
        gate_logits : jnp.ndarray,
        expert_indices : jnp.ndarray,
        num_experts : int,
        top_k : int,
) -> jnp.ndarray:
    """ 
    Compute load balance loss to encourage uniform expert usage

    Args:
        gate_logits : [batch_size, num_experts] - logits from the gate
        expert_indices : [batch_size, top_k] - indices of the top-k experts
        num_experts : Total number of experts
        top_k : Number of experts to consider
    
        Returns:
            loss : Scalar loss value
    """
    batch_size = gate_logits.shape[0]

    # Compute gate probabilities
    gate_probs = jax.nn.softmax(gate_logits, axis = -1) # [batch_size, num_experts]
    mean_gate_prob = jnp.mean(gate_probs, axis = 0) # [num_experts] Average preference over a batch (ideally must be 1/num_experts)

    print(f"Mean gate probability: {mean_gate_prob}")
    print(f"Ideal gate probability: {1 / num_experts}")

    # Compute actual usage for each expert
    expert_usage = jnp.zeros(num_experts)
    for expert_id in range(num_experts):
        usage_count = jnp.sum(expert_indices == expert_id) # Count how many times this expert was selected
        expert_usage = expert_usage.at[expert_id].set(usage_count)

    mean_expert_usage = expert_usage / (batch_size * top_k) # Normalize to [0, 1] # How many times was this expert used on average
    ideal_usage = 1 / num_experts # Ideal usage is 1/num_experts

    print(f"Mean expert usage: {mean_expert_usage}")
    print(f"Ideal expert usage: {ideal_usage}")

    # Load balancing loss
    load_loss = num_experts * jnp.sum(mean_gate_prob*mean_expert_usage)
    # We ideally want to have a low load balancing loss
    print(f"Load balancing loss: {load_loss}")

    return load_loss

class LoadBalancedGate(nn.Module):
    """Complete load balanced gate"""
    num_experts: int
    top_k: int = 2
    load_balance_weight: float = 0.01
    
    @nn.compact
    def __call__(self, x, return_load_loss: bool = True):  # ✅ Add this parameter
        batch_size, seq_len, dim = x.shape
        
        # Same routing logic as before
        routing_input = jnp.mean(x, axis=1)
        gate_hidden = nn.Dense(x.shape[-1], name='gate_hidden')(routing_input)
        gate_hidden = nn.silu(gate_hidden)
        logits = nn.Dense(self.num_experts, name='gate_output')(gate_hidden)
        
        # Top-k selection
        top_k_logits, top_k_indices = jax.lax.top_k(logits, self.top_k)
        top_k_weights = jax.nn.softmax(top_k_logits, axis=-1)
        
        # ✅ Conditional load balancing loss
        if return_load_loss:
            load_loss = compute_load_balance_loss(
                logits, top_k_indices, self.num_experts, self.top_k
            )
            weighted_load_loss = self.load_balance_weight * load_loss
        else:
            weighted_load_loss = 0.0
        
        return top_k_weights, top_k_indices, weighted_load_loss


class CompleteMoE(nn.Module):
    """Complete MoE system - simplified to avoid dynamic slicing"""
    dim: int = 128
    hidden_dim: int = 512
    num_experts: int = 4
    top_k: int = 2
    load_balance_weight: float = 0.01
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        batch_size, seq_len, dim = x.shape
        
        # Get routing decisions with load balancing
        top_k_weights, top_k_indices, load_loss = LoadBalancedGate(
            self.num_experts, self.top_k, self.load_balance_weight
        )(x, return_load_loss=training)
        
        # Apply ALL experts to entire batch, then select
        all_expert_outputs = []
        for i in range(self.num_experts): # self.num_experts is known at compile time
            expert = Expert(self.hidden_dim, self.dim, name=f'expert_{i}')
            expert_out = expert(x)  # Apply to ENTIRE batch at once
            all_expert_outputs.append(expert_out)
        
        # Stack: [num_experts, batch_size, seq_len, dim]
        all_outputs = jnp.stack(all_expert_outputs, axis=0)
        
        # Now do the weighted combination without dynamic slicing
        output = jnp.zeros_like(x)
        for batch_idx in range(batch_size):
            for k_idx in range(self.top_k):
                expert_idx = top_k_indices[batch_idx, k_idx]
                weight = top_k_weights[batch_idx, k_idx]
                
                # ✅ No dynamic slicing - just indexing(gather operation)
                expert_output = all_outputs[expert_idx, batch_idx]  # [seq_len, dim]
                output = output.at[batch_idx].add(weight * expert_output)
        
        if training:
            return output, load_loss
        else:
            return output

def create_train_state(model, key, learning_rate, input_shape):
    """Create a training state for the model"""
    params = model.init(key, jnp.ones(input_shape), training = True)
    optimizer = optax.adam(learning_rate)

    return train_state.TrainState.create(
        apply_fn = model.apply,
        params = params,
        tx = optimizer
    )


def moe_loss_fn(params, model, batch_x, batch_y, apply_fn):
    """Complete loss function with main task and load balancing loss"""
    output, load_loss = apply_fn(params, batch_x, training = True)

    # Main task
    main_loss = jnp.mean((output - batch_y) ** 2)

    total_loss = main_loss + load_loss

    return total_loss, {
        "main_loss": main_loss,
        "load_loss": load_loss,
        "total_loss": total_loss
    }


@jax.jit
def train_step(state, batch_x, batch_y):
    """Single training step for MoE"""
    loss_fn = lambda params: moe_loss_fn(
        params, None, batch_x, batch_y, state.apply_fn
    )
    
    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    
    return state, metrics