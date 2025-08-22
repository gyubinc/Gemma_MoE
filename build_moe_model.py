#!/usr/bin/env python3
"""
Build MoE Model for Qwen-MoE
여러 도메인별 LoRA 어댑터를 합쳐서 top-1 hard gating MoE 모델로 만드는 핵심 스크립트
"""

import argparse
import logging
import sys
import os
import json
import torch
from typing import Dict, Any, List

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from src.configs import domain_manager
from src.utils import validate_environment, print_gpu_memory_summary

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('moe_building.log')
        ]
    )

class MoEModelBuilder:
    """Build MoE model from multiple domain adapters"""
    
    def __init__(self, base_model_name: str = "Qwen/Qwen3-4B-Instruct-2507", device: str = "cuda:0"):
        self.base_model_name = base_model_name
        self.device = device
        self.domains = domain_manager.get_available_domains()
        
    def load_domain_adapters(self, adapters_dir: str) -> Dict[str, str]:
        """Load available domain adapters"""
        available_adapters = {}
        
        for domain in self.domains:
            adapter_path = os.path.join(adapters_dir, domain, "final_adapter")
            if os.path.exists(adapter_path):
                available_adapters[domain] = adapter_path
                logging.info(f"✅ Found adapter for {domain}: {adapter_path}")
            else:
                logging.warning(f"⚠️ No adapter found for {domain}: {adapter_path}")
        
        return available_adapters
    
    def build_moe_model(self, adapters_dir: str, output_dir: str) -> Dict[str, Any]:
        """Build MoE model from domain adapters"""
        logging.info("🔧 Building MoE model...")
        
        # Load available adapters
        available_adapters = self.load_domain_adapters(adapters_dir)
        
        if not available_adapters:
            raise ValueError("No domain adapters found!")
        
        logging.info(f"Found {len(available_adapters)} domain adapters: {list(available_adapters.keys())}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Build MoE configuration
        moe_config = {
            "base_model": self.base_model_name,
            "domains": list(available_adapters.keys()),
            "adapter_paths": available_adapters,
            "gating_strategy": "top-1_hard",
            "expert_selection": "domain_based"
        }
        
        # Save MoE configuration
        config_path = os.path.join(output_dir, "moe_config.json")
        with open(config_path, 'w') as f:
            json.dump(moe_config, f, indent=2)
        
        logging.info(f"📁 MoE configuration saved to: {config_path}")
        
        # Create domain mapping for routing
        domain_mapping = {domain: i for i, domain in enumerate(available_adapters.keys())}
        
        # Save domain mapping
        mapping_path = os.path.join(output_dir, "domain_mapping.json")
        with open(mapping_path, 'w') as f:
            json.dump(domain_mapping, f, indent=2)
        
        logging.info(f"📁 Domain mapping saved to: {mapping_path}")
        
        # Create MoE router
        router_config = {
            "type": "hard_gating",
            "strategy": "top-1",
            "domain_mapping": domain_mapping,
            "fallback_domain": list(available_adapters.keys())[0]  # Use first domain as fallback
        }
        
        router_path = os.path.join(output_dir, "router_config.json")
        with open(router_path, 'w') as f:
            json.dump(router_config, f, indent=2)
        
        logging.info(f"📁 Router configuration saved to: {router_path}")
        
        # Create MoE model info
        model_info = {
            "model_type": "qwen_moe",
            "version": "2.0",
            "base_model": self.base_model_name,
            "num_experts": len(available_adapters),
            "domains": list(available_adapters.keys()),
            "gating_strategy": "top-1_hard",
            "build_time": torch.datetime.now().isoformat(),
            "adapter_paths": available_adapters
        }
        
        info_path = os.path.join(output_dir, "model_info.json")
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        logging.info(f"📁 Model info saved to: {info_path}")
        
        return {
            "output_dir": output_dir,
            "num_experts": len(available_adapters),
            "domains": list(available_adapters.keys()),
            "config_path": config_path,
            "mapping_path": mapping_path,
            "router_path": router_path,
            "info_path": info_path
        }
    
    def validate_moe_model(self, moe_dir: str) -> bool:
        """Validate the built MoE model"""
        logging.info("🔍 Validating MoE model...")
        
        required_files = [
            "moe_config.json",
            "domain_mapping.json", 
            "router_config.json",
            "model_info.json"
        ]
        
        for file in required_files:
            file_path = os.path.join(moe_dir, file)
            if not os.path.exists(file_path):
                logging.error(f"❌ Missing required file: {file_path}")
                return False
        
        # Validate configuration
        try:
            with open(os.path.join(moe_dir, "moe_config.json"), 'r') as f:
                config = json.load(f)
            
            if "domains" not in config or "adapter_paths" not in config:
                logging.error("❌ Invalid MoE configuration")
                return False
            
            # Check if adapter paths exist
            for domain, path in config["adapter_paths"].items():
                if not os.path.exists(path):
                    logging.error(f"❌ Adapter path not found: {path}")
                    return False
            
            logging.info("✅ MoE model validation passed")
            return True
            
        except Exception as e:
            logging.error(f"❌ MoE model validation failed: {e}")
            return False

def main():
    """Main function for building MoE model"""
    parser = argparse.ArgumentParser(description="Build MoE model from domain adapters")
    parser.add_argument("--adapters-dir", default="domain_models",
                       help="Directory containing domain adapters")
    parser.add_argument("--output-dir", default="moe_model",
                       help="Output directory for MoE model")
    parser.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct-2507",
                       help="Base model name")
    parser.add_argument("--device", default="cuda:0",
                       help="Device to use")
    parser.add_argument("--validate-only", action="store_true",
                       help="Only validate existing MoE model")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("🏗️ Starting MoE model building process")
    logger.info(f"Adapters directory: {args.adapters_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Base model: {args.base_model}")
    
    # Validate environment
    if not validate_environment():
        logger.error("❌ Environment validation failed")
        return 1
    
    # Print GPU memory before building
    print_gpu_memory_summary("Before MoE building")
    
    try:
        # Initialize MoE builder
        builder = MoEModelBuilder(args.base_model, args.device)
        
        if args.validate_only:
            # Only validate existing model
            if builder.validate_moe_model(args.output_dir):
                logger.info("✅ MoE model validation completed successfully")
                return 0
            else:
                logger.error("❌ MoE model validation failed")
                return 1
        
        # Build MoE model
        result = builder.build_moe_model(args.adapters_dir, args.output_dir)
        
        logger.info(f"✅ MoE model built successfully!")
        logger.info(f"   Output directory: {result['output_dir']}")
        logger.info(f"   Number of experts: {result['num_experts']}")
        logger.info(f"   Domains: {result['domains']}")
        
        # Validate the built model
        if builder.validate_moe_model(args.output_dir):
            logger.info("✅ MoE model validation passed")
        else:
            logger.error("❌ MoE model validation failed")
            return 1
        
        # Save build summary
        summary = {
            "build_time": torch.datetime.now().isoformat(),
            "result": result,
            "config": {
                "adapters_dir": args.adapters_dir,
                "output_dir": args.output_dir,
                "base_model": args.base_model,
                "device": args.device
            }
        }
        
        summary_path = os.path.join(args.output_dir, "build_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"📊 Build summary saved to: {summary_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ MoE model building failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
