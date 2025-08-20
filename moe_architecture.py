#!/usr/bin/env python3
"""
MoE (Mixture of Experts) Architecture Implementation
Combines 4 domain-specific LoRA adapters into a single MoE model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class Router(nn.Module):
    """Router network for expert selection"""
    
    def __init__(self, hidden_size: int, num_experts: int, router_type: str = "top2"):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.router_type = router_type
        
        # Router network
        self.router = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_experts)
        )
        
        # Load balancing loss weight
        self.load_balancing_loss_weight = 0.01
        
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of router
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            
        Returns:
            router_logits: Expert selection logits
            expert_weights: Expert weights for top-k experts
            expert_indices: Expert indices for top-k experts
        """
        # Get router logits
        router_logits = self.router(hidden_states)  # [batch_size, seq_len, num_experts]
        
        if self.router_type == "top2":
            # Select top-2 experts
            expert_weights, expert_indices = torch.topk(
                F.softmax(router_logits, dim=-1), k=2, dim=-1
            )
        else:
            # Greedy routing (top-1)
            expert_weights = F.softmax(router_logits, dim=-1)
            expert_indices = torch.argmax(router_logits, dim=-1, keepdim=True)
        
        return router_logits, expert_weights, expert_indices
    
    def compute_load_balancing_loss(self, router_logits: torch.Tensor) -> torch.Tensor:
        """Compute load balancing loss to encourage expert utilization"""
        # Compute expert utilization
        expert_usage = torch.mean(F.softmax(router_logits, dim=-1), dim=[0, 1])  # [num_experts]
        
        # Load balancing loss (encourage uniform usage)
        uniform_usage = torch.ones_like(expert_usage) / self.num_experts
        load_balancing_loss = F.mse_loss(expert_usage, uniform_usage)
        
        return load_balancing_loss * self.load_balancing_loss_weight

class MoEModel(nn.Module):
    """Mixture of Experts model combining domain-specific adapters"""
    
    def __init__(
        self, 
        base_model_name: str,
        domain_adapters: Dict[str, str],
        num_experts: int = 4,
        router_type: str = "top2",
        expert_capacity: int = 64
    ):
        super().__init__()
        
        self.base_model_name = base_model_name
        self.domain_adapters = domain_adapters
        self.num_experts = num_experts
        self.router_type = router_type
        self.expert_capacity = expert_capacity
        
        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load domain-specific adapters
        self.experts = nn.ModuleDict()
        self.domain_to_expert = {}
        
        for i, (domain, adapter_path) in enumerate(domain_adapters.items()):
            if i >= num_experts:
                break
                
            # Load adapter
            expert_model = PeftModel.from_pretrained(
                self.base_model, 
                adapter_path,
                is_trainable=False  # Freeze expert weights
            )
            
            self.experts[f"expert_{i}"] = expert_model
            self.domain_to_expert[domain] = f"expert_{i}"
            
            logger.info(f"Loaded {domain} adapter as expert_{i}")
        
        # Initialize router
        hidden_size = self.base_model.config.hidden_size
        self.router = Router(hidden_size, num_experts, router_type)
        
        # Expert capacity
        self.expert_capacity = expert_capacity
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_router_loss: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of MoE model
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Labels for loss computation
            return_router_loss: Whether to include router loss
            
        Returns:
            Dictionary containing outputs and losses
        """
        batch_size, seq_len = input_ids.shape
        
        # Get base model hidden states (without adapters)
        with torch.no_grad():
            base_outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            hidden_states = base_outputs.hidden_states[-1]  # Last layer hidden states
        
        # Router forward pass
        router_logits, expert_weights, expert_indices = self.router(hidden_states)
        
        # Initialize output hidden states
        output_hidden_states = torch.zeros_like(hidden_states)
        
        # Process each expert
        expert_outputs = {}
        router_loss = 0.0
        
        for expert_idx in range(self.num_experts):
            expert_name = f"expert_{expert_idx}"
            if expert_name not in self.experts:
                continue
                
            # Get expert mask
            expert_mask = (expert_indices == expert_idx).any(dim=-1)  # [batch_size, seq_len]
            
            if not expert_mask.any():
                continue
            
            # Apply expert to masked inputs
            masked_input_ids = input_ids[expert_mask]
            masked_attention_mask = attention_mask[expert_mask] if attention_mask is not None else None
            
            # Limit to expert capacity
            if masked_input_ids.shape[0] > self.expert_capacity:
                masked_input_ids = masked_input_ids[:self.expert_capacity]
                masked_attention_mask = masked_attention_mask[:self.expert_capacity] if masked_attention_mask is not None else None
                expert_mask = expert_mask[:self.expert_capacity]
            
            # Forward pass through expert
            with torch.no_grad():
                expert_output = self.experts[expert_name](
                    input_ids=masked_input_ids,
                    attention_mask=masked_attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                )
                expert_hidden = expert_output.hidden_states[-1]
            
            # Store expert output
            expert_outputs[expert_name] = {
                'hidden_states': expert_hidden,
                'mask': expert_mask,
                'weights': expert_weights[expert_mask, :, expert_idx] if expert_mask.any() else None
            }
        
        # Combine expert outputs
        for expert_name, expert_data in expert_outputs.items():
            if expert_data['mask'] is not None and expert_data['mask'].any():
                mask = expert_data['mask']
                expert_hidden = expert_data['hidden_states']
                weights = expert_data['weights']
                
                # Weighted combination
                if weights is not None:
                    weighted_hidden = expert_hidden * weights.unsqueeze(-1)
                else:
                    weighted_hidden = expert_hidden
                
                # Add to output
                output_hidden_states[mask] = weighted_hidden
        
        # Final forward pass with combined hidden states
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            hidden_states=output_hidden_states,
            return_dict=True
        )
        
        # Compute router loss
        if return_router_loss:
            router_loss = self.router.compute_load_balancing_loss(router_logits)
            outputs['router_loss'] = router_loss
            outputs['total_loss'] = outputs['loss'] + router_loss
        
        # Add routing information
        outputs['router_logits'] = router_logits
        outputs['expert_weights'] = expert_weights
        outputs['expert_indices'] = expert_indices
        
        return outputs
    
    def get_expert_utilization(self, router_logits: torch.Tensor) -> Dict[str, float]:
        """Get expert utilization statistics"""
        expert_probs = F.softmax(router_logits, dim=-1)  # [batch_size, seq_len, num_experts]
        expert_usage = torch.mean(expert_probs, dim=[0, 1])  # [num_experts]
        
        utilization = {}
        for i in range(self.num_experts):
            expert_name = f"expert_{i}"
            utilization[expert_name] = expert_usage[i].item()
        
        return utilization

def create_moe_model(
    base_model_name: str,
    domain_adapters: Dict[str, str],
    num_experts: int = 4,
    router_type: str = "top2"
) -> MoEModel:
    """Create MoE model from domain adapters"""
    
    logger.info(f"Creating MoE model with {num_experts} experts")
    logger.info(f"Domain adapters: {list(domain_adapters.keys())}")
    
    return MoEModel(
        base_model_name=base_model_name,
        domain_adapters=domain_adapters,
        num_experts=num_experts,
        router_type=router_type
    )

def load_domain_adapters(domain_models_dir: str) -> Dict[str, str]:
    """Load domain adapter paths"""
    import os
    
    domain_adapters = {}
    domains = ["medical", "law", "math", "code"]
    
    for domain in domains:
        adapter_path = os.path.join(domain_models_dir, domain, "final_adapter")
        if os.path.exists(adapter_path):
            domain_adapters[domain] = adapter_path
            logger.info(f"Found {domain} adapter: {adapter_path}")
        else:
            logger.warning(f"Missing {domain} adapter: {adapter_path}")
    
    return domain_adapters

if __name__ == "__main__":
    # Test MoE model creation
    logging.basicConfig(level=logging.INFO)
    
    # Load domain adapters
    domain_adapters = load_domain_adapters("./domain_models")
    
    if len(domain_adapters) < 2:
        print("❌ Need at least 2 domain adapters to create MoE model")
        exit(1)
    
    # Create MoE model
    moe_model = create_moe_model(
        base_model_name="Qwen/Qwen3-4B-Instruct-2507",
        domain_adapters=domain_adapters,
        num_experts=len(domain_adapters),
        router_type="top2"
    )
    
    print(f"✅ MoE model created with {len(domain_adapters)} experts")
    print(f"📊 Expert utilization: {moe_model.get_expert_utilization(torch.randn(1, 10, moe_model.base_model.config.hidden_size))}")
