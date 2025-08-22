#!/usr/bin/env python3
"""
Data analysis script for Qwen-MoE project
"""

import argparse
import logging
import sys
import os
import json

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.configs import domain_manager
from src.utils import check_data_availability, analyze_dataset_samples, get_domain_statistics

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    """Main analysis function"""
    parser = argparse.ArgumentParser(description="Analyze dataset and domain information")
    parser.add_argument("--domain", 
                       choices=domain_manager.get_available_domains(),
                       help="Domain to analyze (if not specified, analyze all)")
    parser.add_argument("--check-only", action="store_true",
                       help="Only check data availability")
    parser.add_argument("--samples-only", action="store_true",
                       help="Only analyze sample data")
    parser.add_argument("--stats-only", action="store_true",
                       help="Only show statistics")
    parser.add_argument("--max-samples", type=int, default=3,
                       help="Maximum samples to show in analysis")
    parser.add_argument("--output", default="analysis_report.json",
                       help="Output file for analysis report")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        if args.check_only:
            # Only check availability
            logger.info("Checking data availability...")
            availability = check_data_availability([args.domain] if args.domain else None)
            
            print("\n📊 DATA AVAILABILITY CHECK")
            print("="*60)
            for domain, available in availability.items():
                status = "✅ Available" if available else "❌ Missing"
                print(f"{domain.upper():<10}: {status}")
            
            return 0
        
        if args.samples_only:
            # Only analyze samples
            logger.info("Analyzing dataset samples...")
            analyze_dataset_samples([args.domain] if args.domain else None, args.max_samples)
            return 0
        
        if args.stats_only:
            # Only show statistics
            if args.domain:
                logger.info(f"Getting statistics for {args.domain} domain...")
                stats = get_domain_statistics(args.domain)
                print(f"\n📊 {args.domain.upper()} DOMAIN STATISTICS")
                print("="*60)
                for key, value in stats.items():
                    print(f"{key}: {value}")
            else:
                logger.info("Getting statistics for all domains...")
                for domain in domain_manager.get_available_domains():
                    stats = get_domain_statistics(domain)
                    print(f"\n📊 {domain.upper()} DOMAIN STATISTICS")
                    print("="*60)
                    for key, value in stats.items():
                        print(f"{key}: {value}")
            return 0
        
        # Full analysis
        logger.info("Starting comprehensive analysis...")
        
        # Check availability
        availability = check_data_availability([args.domain] if args.domain else None)
        
        # Analyze samples
        analysis = analyze_dataset_samples([args.domain] if args.domain else None, args.max_samples)
        
        # Get statistics
        stats = {}
        domains_to_analyze = [args.domain] if args.domain else domain_manager.get_available_domains()
        for domain in domains_to_analyze:
            stats[domain] = get_domain_statistics(domain)
        
        # Create comprehensive report
        report = {
            "availability": availability,
            "analysis": analysis,
            "statistics": stats
        }
        
        # Save report
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Analysis report saved to: {args.output}")
        
        # Print summary
        print("\n" + "="*80)
        print("📋 ANALYSIS SUMMARY")
        print("="*80)
        
        available_domains = sum(1 for available in availability.values() if available)
        total_domains = len(availability)
        print(f"📊 Data Availability: {available_domains}/{total_domains} domains available")
        
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
