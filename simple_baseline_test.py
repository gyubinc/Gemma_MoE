#!/usr/bin/env python3
"""
Simple baseline evaluation for all domains
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.configs.domains import domain_manager

def load_model():
    """Load base model"""
    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    return model, tokenizer

def evaluate_domain_simple(domain_name, max_samples=50):
    """Simple evaluation for a domain"""
    print(f"Evaluating {domain_name} domain...")
    
    # Load data
    domain_config = domain_manager.get_domain(domain_name)
    test_file = domain_config.get_file_path('test')
    
    with open(test_file, 'r') as f:
        data = json.load(f)
    
    data = data[:max_samples]
    
    # Load model
    model, tokenizer = load_model()
    model.eval()
    
    correct = 0
    total = len(data)
    
    for i, item in enumerate(data):
        if i % 10 == 0:
            print(f"Processing {i}/{total}")
        
        # Format prompt
        if domain_name == "medical":
            question = item.get('question', '')
            options = item.get('options', [])
            choices = "\n".join([f"{chr(65+j)}. {option}" for j, option in enumerate(options)])
            prompt = f"Answer the following medical question. Choose only A, B, C, or D.\n\nQuestion: {question}\n\n{choices}\n\nAnswer:"
            
            # Get correct answer
            correct_option = item.get('correct_option', -1)
            if correct_option == -1:
                correct_answer = item.get('correct_answer', '')
                for j, option in enumerate(options):
                    if option.strip().lower() == correct_answer.strip().lower():
                        correct_option = j
                        break
                if correct_option == -1:
                    correct_option = 0
            reference = chr(65 + correct_option)
            
        elif domain_name == "law":
            context = item.get('context', '')
            endings = item.get('endings', [])
            choices = "\n".join([f"{chr(65+j)}. {ending}" for j, ending in enumerate(endings)])
            prompt = f"Legal case analysis. Choose only A, B, C, D, or E.\n\nContext: {context}\n\nSelect the appropriate holding:\n\n{choices}\n\nAnswer:"
            
            reference = chr(65 + item.get('correct_ending_idx', 0))
            
        elif domain_name == "math":
            question = item.get('question', '')
            options = item.get('options', [])
            choices = "\n".join([f"{chr(65+j)}. {option}" for j, option in enumerate(options)])
            prompt = f"Solve this math problem. Choose only A, B, C, D, or E.\n\nQuestion: {question}\n\n{choices}\n\nAnswer:"
            
            reference = chr(65 + item.get('correct_option', 0))
        
        # Generate
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        prediction = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        
        # Clean prediction
        pred_clean = prediction.upper()
        if len(pred_clean) > 0:
            pred_clean = pred_clean[0]
        
        if pred_clean == reference:
            correct += 1
        
        if i < 3:
            print(f"Sample {i}: pred='{prediction}' -> '{pred_clean}', ref='{reference}', correct={pred_clean == reference}")
    
    accuracy = correct / total if total > 0 else 0.0
    
    # Save results
    result = {
        "domain": domain_name,
        "accuracy": accuracy,
        "total_samples": total,
        "correct_predictions": correct
    }
    
    with open(f"{domain_name}_baseline.json", 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"{domain_name.upper()} accuracy: {accuracy:.4f} ({correct}/{total})")
    return result

def main():
    """Main function"""
    domains = ["medical", "law", "math"]
    
    for domain in domains:
        try:
            evaluate_domain_simple(domain, 50)
        except Exception as e:
            print(f"Error evaluating {domain}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
