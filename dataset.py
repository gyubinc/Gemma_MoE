import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import random
from typing import Dict, List, Tuple, Optional


class DomainDataset(Dataset):
    """Base class for domain-specific datasets"""
    
    def __init__(self, tokenizer, max_length=1024, split='train'):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.data = []
        self.domain_name = "base"
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format as instruction-response
        instruction_text = f"<instruction>: {item['instruction']}\n<response>: "
        response_text = item['response']
        full_text = instruction_text + response_text
        
        # Tokenize instruction and full text separately to get boundaries
        instruction_encoding = self.tokenizer(
            instruction_text,
            add_special_tokens=False
        )
        
        full_encoding = self.tokenizer(
            full_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length
        )
        
        # Create labels: mask instruction part, keep response part
        labels = full_encoding['input_ids'].copy()
        instruction_length = len(instruction_encoding['input_ids'])
        
        # Mask instruction part (set to -100 to ignore in loss calculation)
        labels[:instruction_length] = [-100] * instruction_length
        
        return {
            'input_ids': full_encoding['input_ids'],
            'attention_mask': full_encoding['attention_mask'],
            'labels': labels
        }


class MedicalDataset(DomainDataset):
    """Medical domain dataset using MedMCQA"""
    
    def __init__(self, tokenizer, max_length=1024, split='train'):
        super().__init__(tokenizer, max_length, split)
        self.domain_name = "medical"
        self._load_data()
    
    def _load_data(self):
        try:
            dataset = load_dataset("medmcqa", split=self.split)
            
            for item in dataset:
                question = item['question']
                options = [item['opa'], item['opb'], item['opc'], item['opd']]
                correct_answer = options[item['cop']]
                
                # Format options
                options_text = '\n'.join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
                instruction = f"Question: {question}\n\nOptions:\n{options_text}\n\nAnswer:"
                response = f"{chr(65+item['cop'])}. {correct_answer}"
                
                self.data.append({
                    'instruction': instruction,
                    'response': response
                })
                
        except Exception as e:
            print(f"Error loading MedMCQA dataset: {e}")
            raise RuntimeError(f"Failed to load MedMCQA dataset: {e}. Please ensure the dataset is available.")


class LawDataset(DomainDataset):
    """Law domain dataset using LexGLUE case_hold"""
    
    def __init__(self, tokenizer, max_length=1024, split='train'):
        super().__init__(tokenizer, max_length, split)
        self.domain_name = "law"
        self._load_data()
    
    def _load_data(self):
        try:
            dataset = load_dataset("lex_glue", "case_hold", split=self.split)
            
            for item in dataset:
                context = item['context']
                endings = item['endings']  # 'holdings' -> 'endings'
                correct_idx = item['label']
                
                instruction = f"Legal case analysis:\n{context}\n\nSelect the appropriate holding:"
                response = endings[correct_idx]
                
                self.data.append({
                    'instruction': instruction,
                    'response': response
                })
                
        except Exception as e:
            print(f"Error loading LexGLUE dataset: {e}")
            raise RuntimeError(f"Failed to load LexGLUE case_hold dataset: {e}. Please ensure the dataset is available.")


class MathDataset(DomainDataset):
    """Math domain dataset using GSM8K"""
    
    def __init__(self, tokenizer, max_length=1024, split='train'):
        super().__init__(tokenizer, max_length, split)
        self.domain_name = "math"
        self._load_data()
    
    def _load_data(self):
        try:
            dataset = load_dataset("gsm8k", "main", split=self.split)
            
            for item in dataset:
                question = item['question']
                answer = item['answer']
                
                self.data.append({
                    'instruction': f"Solve this math problem step by step:\n{question}",
                    'response': answer
                })
                
        except Exception as e:
            print(f"Error loading GSM8K dataset: {e}")
            raise RuntimeError(f"Failed to load GSM8K dataset: {e}. Please ensure the dataset is available.")


class CodeDataset(DomainDataset):
    """Code domain dataset using CodeXGLUE"""
    
    def __init__(self, tokenizer, max_length=1024, split='train'):
        super().__init__(tokenizer, max_length, split)
        self.domain_name = "code"
        self._load_data()
    
    def _load_data(self):
        try:
            dataset = load_dataset("code_x_glue_ct_code_to_text", "python", split=self.split)
            
            for item in dataset:
                code = item['code']
                docstring = item['docstring']
                
                self.data.append({
                    'instruction': f"Explain what this Python code does:\n```python\n{code}\n```",
                    'response': docstring
                })
                
        except Exception as e:
            print(f"Error loading CodeXGLUE dataset: {e}")
            raise RuntimeError(f"Failed to load CodeXGLUE dataset: {e}. Please ensure the dataset is available.")


class MultiDomainDataset(Dataset):
    """Combined dataset from multiple domains"""
    
    def __init__(self, datasets: List[DomainDataset], sampling_strategy='balanced'):
        self.datasets = datasets
        self.sampling_strategy = sampling_strategy
        self.data = []
        self._combine_datasets()
    
    def _combine_datasets(self):
        if self.sampling_strategy == 'balanced':
            # Equal samples from each domain
            min_size = min(len(ds) for ds in self.datasets)
            for dataset in self.datasets:
                sampled_indices = random.sample(range(len(dataset)), min_size)
                for idx in sampled_indices:
                    self.data.append((dataset, idx))
        else:
            # Use all data
            for dataset in self.datasets:
                for idx in range(len(dataset)):
                    self.data.append((dataset, idx))
        
        # Shuffle combined data
        random.shuffle(self.data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        dataset, data_idx = self.data[idx]
        return dataset[data_idx]


def create_domain_datasets(tokenizer, max_length=1024, split='train') -> Dict[str, DomainDataset]:
    """Create all domain datasets"""
    datasets = {
        'medical': MedicalDataset(tokenizer, max_length, split),
        'law': LawDataset(tokenizer, max_length, split),
        'math': MathDataset(tokenizer, max_length, split),
        'code': CodeDataset(tokenizer, max_length, split)
    }
    return datasets


def create_dataloaders(datasets: Dict[str, Dataset], batch_size=2, shuffle=True) -> Dict[str, DataLoader]:
    """Create DataLoaders for all datasets"""
    dataloaders = {}
    for name, dataset in datasets.items():
        dataloaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            pin_memory=True
        )
    return dataloaders


def get_evaluation_datasets(tokenizer, max_length=1024):
    """Get evaluation datasets for all domains"""
    eval_datasets = {}
    
    try:
        eval_datasets['medical'] = MedicalDataset(tokenizer, max_length, split='validation')
    except:
        eval_datasets['medical'] = MedicalDataset(tokenizer, max_length, split='train')
    
    try:
        eval_datasets['law'] = LawDataset(tokenizer, max_length, split='validation')
    except:
        eval_datasets['law'] = LawDataset(tokenizer, max_length, split='train')
    
    try:
        eval_datasets['math'] = MathDataset(tokenizer, max_length, split='test')
    except:
        eval_datasets['math'] = MathDataset(tokenizer, max_length, split='train')
    
    try:
        eval_datasets['code'] = CodeDataset(tokenizer, max_length, split='validation')
    except:
        eval_datasets['code'] = CodeDataset(tokenizer, max_length, split='train')
    
    return eval_datasets


def collate_fn(batch):
    """Custom collate function for batching"""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    domains = [item['domain'] for item in batch]
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'domains': domains
    }


if __name__ == "__main__":
    # Test the datasets
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create datasets
    datasets = create_domain_datasets(tokenizer, max_length=512, split='train')
    
    # Print dataset sizes
    for name, dataset in datasets.items():
        print(f"{name}: {len(dataset)} samples")
        
        # Show a sample
        sample = dataset[0]
        print(f"Sample from {name}:")
        print(f"Domain: {sample['domain']}")
        print(f"Input shape: {sample['input_ids'].shape}")
        print("=" * 50)
    
    # Test multi-domain dataset
    multi_dataset = MultiDomainDataset(list(datasets.values()))
    print(f"Multi-domain dataset: {len(multi_dataset)} samples")
    
    # Test dataloader
    dataloader = DataLoader(multi_dataset, batch_size=2, collate_fn=collate_fn)
    for batch in dataloader:
        print(f"Batch input_ids shape: {batch['input_ids'].shape}")
        print(f"Batch domains: {batch['domains']}")
        break
