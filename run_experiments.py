#!/usr/bin/env python3
"""
Experiment runner for systematic attention mechanism comparison.
"""
import subprocess
import json
import time
from itertools import product

def run_experiment(config):
    """Run a single experiment configuration."""
    cmd = ["python", "train.py"]
    
    # Add all configuration parameters
    for key, value in config.items():
        if isinstance(value, bool) and value:
            cmd.append(f"--{key}")
        elif not isinstance(value, bool):
            cmd.extend([f"--{key}", str(value)])
    
    print(f"Running: {' '.join(cmd)}")
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        end_time = time.time()
        
        return {
            "config": config,
            "success": result.returncode == 0,
            "runtime": end_time - start_time,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    except subprocess.TimeoutExpired:
        return {
            "config": config,
            "success": False,
            "runtime": 3600,
            "error": "Timeout"
        }

def scale_experiments():
    """Run experiments across different model scales."""
    print("=== Scale Experiments ===")
    
    configs = []
    
    # Test different model sizes with best-performing attention mechanisms
    model_presets = ['tiny', 'small', 'medium']
    attention_types = ['MultiHeadLatentAttention', 'FlashAttention3', 'CausalSelfAttention']
    
    for preset, attention in product(model_presets, attention_types):
        config = {
            'model_preset': preset,
            'attention': attention,
            'feedforward': 'FeedForwardSwiGLU',  # Best from your results
            'norm': 'RMSNorm',
            'max_iters': 1000,  # Shorter for comparison
            'track_memory': True,
            'save_curves': True,
            'benchmark_inference': True
        }
        configs.append(config)
    
    results = []
    for config in configs:
        result = run_experiment(config)
        results.append(result)
        
        # Save incremental results
        with open('scale_experiment_results.json', 'w') as f:
            json.dump(results, f, indent=2)
    
    return results

def latent_dimension_sweep():
    """Test different latent dimensions for MLA."""
    print("=== MLA Latent Dimension Sweep ===")
    
    configs = []
    base_config = {
        'attention': 'MultiHeadLatentAttention',
        'feedforward': 'FeedForwardSwiGLU',
        'norm': 'RMSNorm',
        'n_embd': 384,
        'max_iters': 1000,
        'track_memory': True,
        'save_curves': True
    }
    
    # Test different latent dimensions
    latent_dims = [48, 96, 192, 384]  # n_embd//8, n_embd//4, n_embd//2, n_embd
    
    for latent_dim in latent_dims:
        config = base_config.copy()
        config['mla_latent_dim'] = latent_dim
        configs.append(config)
    
    results = []
    for config in configs:
        result = run_experiment(config)
        results.append(result)
        
        with open('mla_latent_sweep_results.json', 'w') as f:
            json.dump(results, f, indent=2)
    
    return results

def hip_parameter_sweep():
    """Test different HiP parameters."""
    print("=== HiP Parameter Sweep ===")
    
    configs = []
    base_config = {
        'attention': 'HiPAttention',
        'feedforward': 'FeedForwardGeGLU',  # Best for HiP from your results
        'norm': 'RMSNorm',
        'max_iters': 500,  # Shorter since HiP is slow
        'track_memory': True,
        'save_curves': True
    }
    
    # Test different parameter combinations
    chunk_sizes = [16, 32, 64]
    top_k_chunks = [4, 8, 16]
    
    for chunk_size, top_k in product(chunk_sizes, top_k_chunks):
        config = base_config.copy()
        config['hip_chunk_size'] = chunk_size
        config['hip_top_k_chunks'] = top_k
        configs.append(config)
    
    results = []
    for config in configs:
        result = run_experiment(config)
        results.append(result)
        
        with open('hip_parameter_sweep_results.json', 'w') as f:
            json.dump(results, f, indent=2)
    
    return results

def sequence_length_experiments():
    """Test performance across different sequence lengths."""
    print("=== Sequence Length Experiments ===")
    
    configs = []
    attention_types = ['CausalSelfAttention', 'MultiHeadLatentAttention', 'FlashAttention3']
    block_sizes = [128, 256, 512]
    
    for attention, block_size in product(attention_types, block_sizes):
        config = {
            'attention': attention,
            'feedforward': 'FeedForwardSwiGLU',
            'norm': 'RMSNorm',
            'block_size': block_size,
            'max_iters': 500,
            'batch_size': 32,  # Smaller batch for longer sequences
            'track_memory': True,
            'benchmark_inference': True
        }
        configs.append(config)
    
    results = []
    for config in configs:
        result = run_experiment(config)
        results.append(result)
        
        with open('sequence_length_results.json', 'w') as f:
            json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run systematic experiments')
    parser.add_argument('--experiment', type=str, required=True,
                        choices=['scale', 'mla-latent', 'hip-params', 'sequence-length', 'all'],
                        help='Type of experiment to run')
    
    args = parser.parse_args()
    
    if args.experiment == 'scale':
        scale_experiments()
    elif args.experiment == 'mla-latent':
        latent_dimension_sweep()
    elif args.experiment == 'hip-params':
        hip_parameter_sweep()
    elif args.experiment == 'sequence-length':
        sequence_length_experiments()
    elif args.experiment == 'all':
        scale_experiments()
        latent_dimension_sweep()
        hip_parameter_sweep()
        sequence_length_experiments()
    
    print("Experiments completed!")