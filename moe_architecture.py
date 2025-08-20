#!/usr/bin/env python3
"""
MoE Architecture with 5 MLPs per layer (1 original + 4 domain-specific)
Top-1 routing implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GemmaForCausalLM, AutoModelForCausalLM
from peft import PeftModel
import copy
import logging
import os
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class TopKRouter(nn.Module):
    """Top-1 Router for MoE"""
    
    def __init__(self, hidden_size: int, num_experts: int = 5):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        
        # Router network
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        
        # Initialize router weights to be near uniform
        with torch.no_grad():
            self.router.weight.fill_(0.01)
    
    def forward(self, hidden_states: torch.Tensor):
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
        Returns:
            router_logits: [batch_size, seq_len, num_experts]
            selected_experts: [batch_size, seq_len] - expert indices
            expert_weights: [batch_size, seq_len] - weights for selected experts
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Compute router logits
        router_logits = self.router(hidden_states)  # [batch, seq, num_experts]
        
        # Top-1 routing
        expert_weights, selected_experts = torch.topk(router_logits, k=1, dim=-1)
        expert_weights = F.softmax(expert_weights, dim=-1)  # [batch, seq, 1]
        selected_experts = selected_experts.squeeze(-1)  # [batch, seq]
        expert_weights = expert_weights.squeeze(-1)  # [batch, seq]
        
        return router_logits, selected_experts, expert_weights


class MoEMLP(nn.Module):
    """MoE MLP with 5 experts (1 original + 4 domain-specific LoRA adapters)"""
    
    def __init__(self, 
                 original_mlp: nn.Module,
                 domain_lora_adapters: Dict[str, nn.Module],
                 hidden_size: int,
                 aux_loss_weight: float = 0.01):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.aux_loss_weight = aux_loss_weight
        
        # Store original MLP and LoRA adapters
        self.original_mlp = original_mlp
        self.domain_lora_adapters = nn.ModuleDict(domain_lora_adapters)
        
        # Expert order: original, medical, law, math, code
        self.expert_names = ['original'] + list(domain_lora_adapters.keys())
        self.num_experts = len(self.expert_names)
        
        # Router
        self.router = TopKRouter(hidden_size, self.num_experts)
        
        logger.info(f"🔀 MoE MLP initialized with {self.num_experts} experts: {self.expert_names}")
        logger.info(f"   - Original MLP + {len(domain_lora_adapters)} LoRA adapters")
    
    def forward(self, hidden_states: torch.Tensor):
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
        Returns:
            output: [batch_size, seq_len, hidden_size]
            aux_loss: scalar tensor for load balancing
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Get routing decisions
        router_logits, selected_experts, expert_weights = self.router(hidden_states)
        
        # Initialize output
        output = torch.zeros_like(hidden_states)
        
        # Process each expert
        for expert_idx, expert_name in enumerate(self.expert_names):
            # Find tokens routed to this expert
            expert_mask = (selected_experts == expert_idx)  # [batch, seq]
            
            if expert_mask.any():
                # Extract tokens for this expert
                expert_tokens = hidden_states[expert_mask]  # [num_tokens, hidden_size]
                expert_token_weights = expert_weights[expert_mask]  # [num_tokens]
                
                if expert_tokens.numel() > 0:
                    if expert_name == 'original':
                        # Use original MLP
                        expert_output = self.original_mlp(expert_tokens)
                    else:
                        # Use base MLP + LoRA adapter
                        base_output = self.original_mlp(expert_tokens)
                        lora_adapter = self.domain_lora_adapters[expert_name]
                        lora_output = lora_adapter(expert_tokens)
                        expert_output = base_output + lora_output
                    
                    # Apply expert weights
                    expert_output = expert_output * expert_token_weights.unsqueeze(-1)
                    
                    # Place back in output
                    output[expert_mask] = expert_output
        
        # Compute auxiliary load balancing loss
        aux_loss = self._compute_aux_loss(router_logits, selected_experts)
        
        # Store aux loss for trainer to access
        self._last_aux_loss = aux_loss
        
        return output, aux_loss
    
    def _compute_aux_loss(self, router_logits: torch.Tensor, selected_experts: torch.Tensor):
        """Compute auxiliary load balancing loss"""
        # Convert to probabilities
        router_probs = F.softmax(router_logits, dim=-1)  # [batch, seq, num_experts]
        
        # Average probability for each expert
        expert_probs = router_probs.mean(dim=(0, 1))  # [num_experts]
        
        # Count how many tokens are assigned to each expert
        expert_counts = torch.zeros(self.num_experts, device=selected_experts.device)
        for i in range(self.num_experts):
            expert_counts[i] = (selected_experts == i).float().mean()
        
        # Load balancing loss: minimize variance in expert usage
        aux_loss = (expert_probs * expert_counts).sum() * self.num_experts
        
        return aux_loss * self.aux_loss_weight
    
    def get_routing_stats(self, hidden_states: torch.Tensor):
        """Get routing statistics for analysis"""
        with torch.no_grad():
            router_logits, selected_experts, expert_weights = self.router(hidden_states)
            
            # Count tokens per expert
            expert_counts = torch.zeros(self.num_experts)
            total_tokens = selected_experts.numel()
            
            for i in range(self.num_experts):
                expert_counts[i] = (selected_experts == i).sum().item()
            
            expert_ratios = expert_counts / total_tokens
            
            stats = {}
            for i, expert_name in enumerate(self.expert_names):
                stats[expert_name] = {
                    'count': expert_counts[i].item(),
                    'ratio': expert_ratios[i].item()
                }
            
            return stats


class MoEGemmaModel(nn.Module):
    """Gemma model with MoE MLPs"""
    
    def __init__(self, 
                 base_model_path: str,
                 domain_adapter_paths: Dict[str, str],
                 aux_loss_weight: float = 0.01):
        super().__init__()
        
        # Load base model
        logger.info(f"📚 Loading base model from {base_model_path}")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        
        # Load domain LoRA adapters and extract MLP LoRA layers
        domain_lora_adapters_by_layer = {}
        
        for domain, adapter_path in domain_adapter_paths.items():
            logger.info(f"📚 Loading {domain} LoRA adapter from {adapter_path}")
            
            # Load LoRA model
            domain_model = PeftModel.from_pretrained(
                AutoModelForCausalLM.from_pretrained(
                    base_model_path,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                ),
                adapter_path
            )
            
            # Extract LoRA adapters for MLP layers from each transformer layer
            for layer_idx in range(len(domain_model.base_model.model.layers)):
                if layer_idx not in domain_lora_adapters_by_layer:
                    domain_lora_adapters_by_layer[layer_idx] = {}
                
                # Extract LoRA adapters for this layer's MLP
                layer = domain_model.base_model.model.layers[layer_idx]
                mlp_lora_adapters = {}
                
                # Extract gate_proj, up_proj, down_proj LoRA adapters
                if hasattr(layer.mlp.gate_proj, 'lora_A'):
                    mlp_lora_adapters['gate_proj'] = layer.mlp.gate_proj
                if hasattr(layer.mlp.up_proj, 'lora_A'):
                    mlp_lora_adapters['up_proj'] = layer.mlp.up_proj
                if hasattr(layer.mlp.down_proj, 'lora_A'):
                    mlp_lora_adapters['down_proj'] = layer.mlp.down_proj
                
                domain_lora_adapters_by_layer[layer_idx][domain] = mlp_lora_adapters
            
            # Clean up domain model to save memory
            del domain_model
            torch.cuda.empty_cache()
        
        # Replace MLPs with MoE MLPs
        logger.info("🔄 Replacing MLPs with MoE MLPs...")
        hidden_size = self.base_model.config.hidden_size
        
        for layer_idx in range(len(self.base_model.model.layers)):
            original_mlp = self.base_model.model.layers[layer_idx].mlp
            domain_lora_adapters = domain_lora_adapters_by_layer.get(layer_idx, {})
            
            # Create MoE MLP for this layer
            moe_mlp = MoEMLP(
                original_mlp=original_mlp,
                domain_lora_adapters=domain_lora_adapters,
                hidden_size=hidden_size,
                aux_loss_weight=aux_loss_weight
            )
            
            # Replace the original MLP
            self.base_model.model.layers[layer_idx].mlp = moe_mlp
        
        logger.info(f"✅ MoE Gemma model created with {len(self.base_model.model.layers)} MoE layers")
    
    def forward(self, *args, **kwargs):
        """Forward pass through MoE model"""
        return self.base_model(*args, **kwargs)
    
    def generate(self, *args, **kwargs):
        """Generation with MoE model"""
        return self.base_model.generate(*args, **kwargs)
    
    def get_routing_stats(self, input_ids: torch.Tensor):
        """Get routing statistics for all layers"""
        with torch.no_grad():
            # Get hidden states for each layer
            outputs = self.base_model(input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            
            layer_stats = {}
            for layer_idx, layer_hidden_states in enumerate(hidden_states[1:]):  # Skip embedding layer
                moe_mlp = self.base_model.model.layers[layer_idx].mlp
                if isinstance(moe_mlp, MoEMLP):
                    layer_stats[f'layer_{layer_idx}'] = moe_mlp.get_routing_stats(layer_hidden_states)
            
            return layer_stats
    
    def save_pretrained(self, save_directory: str):
        """Save MoE model"""
        os.makedirs(save_directory, exist_ok=True)
        self.base_model.save_pretrained(save_directory)
        logger.info(f"💾 MoE model saved to {save_directory}")
    
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """Load pre-trained MoE model"""
        base_model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        
        # Create a new instance and set the base model
        instance = cls.__new__(cls)
        instance.base_model = base_model
        
        return instance


def create_moe_model(base_model_path: str, 
                    domain_adapter_paths: Dict[str, str],
                    aux_loss_weight: float = 0.01) -> MoEGemmaModel:
    """Create MoE Gemma model from base model and domain LoRA adapters"""
    
    logger.info("🚀 Creating LoRA-based MoE Gemma model...")
    logger.info(f"📚 Base model: {base_model_path}")
    logger.info(f"🎯 Domain LoRA adapters: {list(domain_adapter_paths.keys())}")
    
    moe_model = MoEGemmaModel(
        base_model_path=base_model_path,
        domain_adapter_paths=domain_adapter_paths,
        aux_loss_weight=aux_loss_weight
    )
    
    logger.info("✅ LoRA-based MoE model creation completed!")
    return moe_model
