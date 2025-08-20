import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import json
import logging
from pathlib import Path
from sklearn.metrics import accuracy_score
import evaluate
from transformers import AutoTokenizer
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_logging(log_file: Optional[str] = None):
    """Setup logging configuration"""
    if log_file:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )


def print_gpu_memory_summary(stage: str = "", gpu_id: int = 0):
    """Print GPU memory usage summary in the requested format"""
    if torch.cuda.is_available():
        # Get memory stats in bytes for specific GPU
        alloc = torch.cuda.memory_allocated(gpu_id) / (1024**3)  # Convert to GiB
        reserved = torch.cuda.memory_reserved(gpu_id) / (1024**3)
        max_alloc = torch.cuda.max_memory_allocated(gpu_id) / (1024**3)
        
        # Get total GPU memory
        total = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)
        free = total - reserved
        
        stage_prefix = f"[{stage}] " if stage else ""
        print(f"{stage_prefix}[GPU] mem: alloc={alloc:.2f} GiB, reserved={reserved:.2f} GiB, "
              f"max_alloc={max_alloc:.2f} GiB, free={free:.2f} GiB/ total={total:.2f} GiB")
    else:
        print("[GPU] CUDA not available")


def save_config(config: Dict, save_path: str):
    """Save configuration to JSON file"""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Configuration saved to {save_path}")


def load_config(config_path: str) -> Dict:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    logger.info(f"Configuration loaded from {config_path}")
    return config


def calculate_model_parameters(model):
    """Calculate total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")
    
    return total_params, trainable_params


def plot_router_usage(router_stats: Dict[str, Dict], save_path: Optional[str] = None):
    """Plot router LoRA adapter usage statistics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    layer_indices = list(router_stats.keys())
    num_layers_to_plot = min(4, len(layer_indices))
    
    for i in range(num_layers_to_plot):
        layer_key = layer_indices[i]
        adapter_usage = router_stats[layer_key]['adapter_usage']
        
        axes[i].bar(range(len(adapter_usage)), adapter_usage)
        axes[i].set_title(f'Router Usage - {layer_key}')
        axes[i].set_xlabel('LoRA Adapter Index')
        axes[i].set_ylabel('Usage Probability')
        axes[i].set_xticks(range(len(adapter_usage)))
        axes[i].set_xticklabels(['Medical', 'Law', 'Math', 'Code'][:len(adapter_usage)])
    
    # Hide unused subplots
    for i in range(num_layers_to_plot, 4):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Router usage plot saved to {save_path}")
    
    plt.show()


def evaluate_medical_accuracy(predictions: List[str], references: List[str]) -> float:
    """Evaluate accuracy for medical domain (MedMCQA)"""
    correct = 0
    total = len(predictions)
    
    for pred, ref in zip(predictions, references):
        # Extract answer choice (A, B, C, D) from prediction
        pred_choice = extract_choice(pred)
        ref_choice = extract_choice(ref)
        
        if pred_choice and ref_choice and pred_choice == ref_choice:
            correct += 1
    
    accuracy = correct / total if total > 0 else 0.0
    logger.info(f"Medical accuracy: {accuracy:.4f} ({correct}/{total})")
    return accuracy


def evaluate_law_accuracy(predictions: List[str], references: List[str]) -> float:
    """Evaluate accuracy for law domain (case_hold)"""
    # For case_hold, we compare the generated text with reference
    correct = 0
    total = len(predictions)
    
    for pred, ref in zip(predictions, references):
        # Simple string similarity check
        if pred.strip().lower() in ref.strip().lower() or ref.strip().lower() in pred.strip().lower():
            correct += 1
    
    accuracy = correct / total if total > 0 else 0.0
    logger.info(f"Law accuracy: {accuracy:.4f} ({correct}/{total})")
    return accuracy


def evaluate_math_exact_match(predictions: List[str], references: List[str]) -> float:
    """Evaluate exact match for math domain (GSM8K)"""
    correct = 0
    total = len(predictions)
    
    for pred, ref in zip(predictions, references):
        # Extract numerical answer from both prediction and reference
        pred_answer = extract_numerical_answer(pred)
        ref_answer = extract_numerical_answer(ref)
        
        if pred_answer is not None and ref_answer is not None:
            if abs(pred_answer - ref_answer) < 1e-6:  # Handle floating point precision
                correct += 1
    
    exact_match = correct / total if total > 0 else 0.0
    logger.info(f"Math exact match (GSM8K): {exact_match:.4f} ({correct}/{total})")
    return exact_match


def evaluate_code_bleu(predictions: List[str], references: List[str]) -> float:
    """Evaluate BLEU score for code domain"""
    try:
        bleu = evaluate.load("bleu")
        
        # Prepare references in the format expected by evaluate
        references_formatted = [[ref] for ref in references]
        
        results = bleu.compute(
            predictions=predictions,
            references=references_formatted
        )
        
        bleu_score = results['bleu']
        logger.info(f"Code BLEU score: {bleu_score:.4f}")
        return bleu_score
        
    except Exception as e:
        logger.warning(f"Failed to compute BLEU score: {e}")
        return 0.0


def extract_choice(text: str) -> Optional[str]:
    """Extract choice (A, B, C, D) from text"""
    # Look for pattern like "A." or "A:" or just "A"
    pattern = r'\b([A-D])\b'
    match = re.search(pattern, text.upper())
    return match.group(1) if match else None


def extract_numerical_answer(text: str) -> Optional[float]:
    """Extract numerical answer from text"""
    # Look for numbers in the text (including decimals)
    pattern = r'-?\d+\.?\d*'
    matches = re.findall(pattern, text)
    
    if matches:
        try:
            # Return the last number found (often the final answer)
            return float(matches[-1])
        except ValueError:
            return None
    return None


def format_training_stats(stats: Dict) -> str:
    """Format training statistics for logging"""
    formatted = []
    for key, value in stats.items():
        if isinstance(value, float):
            formatted.append(f"{key}: {value:.4f}")
        else:
            formatted.append(f"{key}: {value}")
    return " | ".join(formatted)


def save_router_weights(model, save_path: str):
    """Save only router weights"""
    router_state_dict = {}
    
    for name, param in model.named_parameters():
        if 'router' in name:
            router_state_dict[name] = param.cpu()
    
    torch.save(router_state_dict, save_path)
    logger.info(f"Router weights saved to {save_path}")


def load_router_weights(model, weights_path: str):
    """Load router weights"""
    router_state_dict = torch.load(weights_path, map_location='cpu')
    
    # Load only router parameters
    for name, param in model.named_parameters():
        if name in router_state_dict:
            param.data.copy_(router_state_dict[name])
    
    logger.info(f"Router weights loaded from {weights_path}")


def get_lr_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int):
    """Get learning rate scheduler with warmup"""
    from transformers import get_cosine_schedule_with_warmup
    
    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )


def compute_loss_with_aux(lm_loss: torch.Tensor, aux_loss: torch.Tensor, aux_weight: float = 0.01) -> torch.Tensor:
    """Compute total loss with auxiliary loss"""
    total_loss = lm_loss + aux_weight * aux_loss
    return total_loss


def prepare_inputs_for_generation(tokenizer, text: str, max_length: int = 1024) -> Dict[str, torch.Tensor]:
    """Prepare inputs for text generation"""
    encoding = tokenizer(
        text,
        return_tensors='pt',
        max_length=max_length,
        truncation=True,
        padding=True
    )
    return encoding


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 100, temperature: float = 0.7) -> str:
    """Generate text using the model"""
    inputs = prepare_inputs_for_generation(tokenizer, prompt)
    
    with torch.no_grad():
        # Prefer model.generate if available; fallback to base_model.generate for wrappers
        if hasattr(model, 'generate'):
            gen_target = model
        else:
            gen_target = getattr(model, 'base_model', model)
        # Resolve device from target with parameters
        try:
            device = next(gen_target.parameters()).device
        except StopIteration:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        outputs = gen_target.generate(
            input_ids=inputs['input_ids'].to(device),
            attention_mask=inputs['attention_mask'].to(device),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode only the new tokens
    new_tokens = outputs[0][len(inputs['input_ids'][0]):]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    return generated_text


def create_domain_evaluation_prompts() -> Dict[str, List[str]]:
    """Create evaluation prompts for each domain"""
    prompts = {
        'medical': [
            "A patient presents with sudden chest pain. What is the first test that should be performed in the emergency room?\nA. Chest X-ray\nB. Electrocardiogram (ECG)\nC. Blood test\nD. CT scan\nAnswer:",
            "What is the target HbA1c level for blood glucose management in diabetic patients?\nAnswer:"
        ],
        'law': [
            "What are the legal grounds for canceling a contract after signing?\nAnswer:",
            "Explain the rights and obligations of tenants in a lease agreement.\nAnswer:"
        ],
        'math': [
            "In a right triangle, if one side is 3 and another side is 4, what is the length of the hypotenuse?\nAnswer:",
            "What is the area of a circle with radius 5 cm?\nAnswer:"
        ],
        'code': [
            "Explain what this Python code does:\n```python\ndef binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1\n```\nAnswer:",
            "Write code using list comprehension to generate even numbers from 1 to 10.\nAnswer:"
        ]
    }
    return prompts


def evaluate_all_domains(model, tokenizer, save_results: bool = True, results_path: str = "evaluation_results.json"):
    """Evaluate model on all domains"""
    prompts = create_domain_evaluation_prompts()
    results = {}
    
    print_gpu_memory_summary()
    
    for domain, domain_prompts in prompts.items():
        print(f"\n=== Evaluating {domain.upper()} domain ===")
        domain_results = []
        
        for prompt in domain_prompts:
            generated = generate_text(model, tokenizer, prompt, max_new_tokens=100)
            domain_results.append({
                'prompt': prompt,
                'generated': generated
            })
            print(f"Prompt: {prompt[:100]}...")
            print(f"Generated: {generated[:200]}...")
            print("-" * 50)
        
        results[domain] = domain_results
    
    if save_results:
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Evaluation results saved to {results_path}")
    
    return results


if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    # Test GPU memory
    print_gpu_memory_summary()
    
    # Test choice extraction
    test_text = "The answer is B. This is the correct choice."
    choice = extract_choice(test_text)
    print(f"Extracted choice: {choice}")
    
    # Test numerical extraction
    test_math = "Step 1: 3 + 4 = 7. Step 2: 7 * 2 = 14. The final answer is 14."
    number = extract_numerical_answer(test_math)
    print(f"Extracted number: {number}")
    
    # Test evaluation prompts
    prompts = create_domain_evaluation_prompts()
    for domain, domain_prompts in prompts.items():
        print(f"{domain}: {len(domain_prompts)} prompts")
    
    print("All tests completed!")
