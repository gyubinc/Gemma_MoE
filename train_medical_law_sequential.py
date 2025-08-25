#!/usr/bin/env python3
"""
Sequential training script for Medical and Law domains
"""

import subprocess
import sys
import os
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_training(domain, max_samples, epochs, batch_size, learning_rate, output_dir):
    """Run training for a specific domain"""
    logger.info(f"🚀 Starting training for {domain} domain")
    logger.info(f"Configuration: max_samples={max_samples}, epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")
    
    cmd = [
        "python", "train_domain_lora.py",
        "--domain", domain,
        "--max-samples", str(max_samples),
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--learning-rate", str(learning_rate),
        "--output-dir", output_dir
    ]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"✅ {domain} training completed successfully!")
        logger.info(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {domain} training failed!")
        logger.error(f"Error: {e.stderr}")
        return False

def main():
    """Main training sequence"""
    logger.info("🎯 Starting sequential training: Medical → Law")
    
    # Training configuration
    config = {
        "medical": {
            "max_samples": 20000,
            "epochs": 3,
            "batch_size": 2,
            "learning_rate": 0.0002,
            "output_dir": "domain_models/medical"
        },
        "law": {
            "max_samples": 20000,
            "epochs": 3,
            "batch_size": 2,
            "learning_rate": 0.0002,
            "output_dir": "domain_models/law"
        }
    }
    
    # Train Medical domain
    logger.info("=" * 50)
    logger.info("🏥 TRAINING MEDICAL DOMAIN")
    logger.info("=" * 50)
    
    medical_success = run_training(
        domain="medical",
        **config["medical"]
    )
    
    if not medical_success:
        logger.error("❌ Medical training failed. Stopping sequence.")
        return False
    
    # Train Law domain
    logger.info("=" * 50)
    logger.info("⚖️ TRAINING LAW DOMAIN")
    logger.info("=" * 50)
    
    law_success = run_training(
        domain="law",
        **config["law"]
    )
    
    if not law_success:
        logger.error("❌ Law training failed.")
        return False
    
    logger.info("=" * 50)
    logger.info("🎉 ALL TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("=" * 50)
    logger.info("📁 Models saved to:")
    logger.info(f"  Medical: {config['medical']['output_dir']}")
    logger.info(f"  Law: {config['law']['output_dir']}")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
