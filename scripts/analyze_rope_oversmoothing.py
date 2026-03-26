"""
RoPE Over-Smoothing Analysis for Rebuttal
==========================================

Experiments:
1. MAD (Mean Average Distance) across layers with/without RoPE
   - Higher MAD = less over-smoothing = more distinguishable embeddings
2. Performance vs number of layers with/without RoPE
3. Cosine similarity of node embeddings across layers (smoothing indicator)

Usage:
    python analyze_rope_oversmoothing.py \
        --config configs/gowalla_high_mem.yaml \
        --vis_checkpoint ./outModels/gowalla_high_mem_best.mod \
        --vis_output_dir ./rebuttal/reviewer2/real_results/rope_analysis/gowalla
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import defaultdict

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif', 'Liberation Serif', 'FreeSerif', 'serif']
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['axes.unicode_minus'] = False

from Params import args
from DataHandler import DataHandler
from HiMoE import HiMoE_Model, apply_rotary_encoding


def compute_mad(embeddings, sample_size=5000):
    """
    Compute Mean Average Distance (MAD) of node embeddings.
    Higher MAD = more distinguishable = less over-smoothing.
    """
    N = embeddings.shape[0]
    if N > sample_size:
        idx = np.random.choice(N, sample_size, replace=False)
        emb = embeddings[idx]
    else:
        emb = embeddings
    
    # Pairwise cosine similarity
    emb_norm = F.normalize(emb, p=2, dim=1)
    sim_matrix = torch.mm(emb_norm, emb_norm.t())  # [n, n]
    
    # MAD = mean of (1 - cosine_similarity) for all pairs
    n = emb_norm.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=sim_matrix.device)
    mad = (1 - sim_matrix[mask]).mean().item()
    
    return mad


def compute_embedding_stats(embeddings, sample_size=5000):
    """Compute various embedding quality metrics."""
    N = embeddings.shape[0]
    if N > sample_size:
        idx = np.random.choice(N, sample_size, replace=False)
        emb = embeddings[idx]
    else:
        emb = embeddings
    
    emb_norm = F.normalize(emb, p=2, dim=1)
    sim_matrix = torch.mm(emb_norm, emb_norm.t())
    n = emb_norm.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=sim_matrix.device)
    
    avg_cosine_sim = sim_matrix[mask].mean().item()
    mad = (1 - sim_matrix[mask]).mean().item()
    embedding_norm = emb.norm(dim=1).mean().item()
    embedding_std = emb.std(dim=0).mean().item()
    
    return {
        'mad': mad,
        'avg_cosine_sim': avg_cosine_sim,
        'embedding_norm': embedding_norm,
        'embedding_std': embedding_std,
    }


def analyze_rope_effect(model, handler, output_dir):
    """
    Compare embedding quality at each layer WITH and WITHOUT RoPE.
    Uses the same model but toggles RoPE in forward pass.
    """
    print("\n" + "="*70)
    print("RoPE Over-Smoothing Analysis")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    
    num_layers = len(model.moe_layers)
    adj = handler.adj_stu
    
    results = {'with_rope': {}, 'without_rope': {}}
    
    for use_rope, label in [(True, 'with_rope'), (False, 'without_rope')]:
        print(f"\n--- {'With' if use_rope else 'Without'} RoPE ---")
        
        with torch.no_grad():
            if args.use_lrbs:
                embeddings_0 = model.lrbs_embedding(apply_mask=True)
            else:
                embeddings_0 = model.full_embedding.weight
            
            current_emb = embeddings_0
            layer_stats = {}
            
            # Layer 0 (initial embeddings)
            stats = compute_embedding_stats(current_emb[:args.user])
            layer_stats['layer_0'] = stats
            print(f"  Layer 0: MAD={stats['mad']:.4f}, AvgCosSim={stats['avg_cosine_sim']:.4f}, "
                  f"Norm={stats['embedding_norm']:.4f}, Std={stats['embedding_std']:.6f}")
            
            for layer_idx, moe_layer in enumerate(model.moe_layers):
                # MoE aggregation (same for both)
                agg_emb, _ = moe_layer(adj, current_emb)
                
                # Apply or skip RoPE
                if use_rope:
                    rotated_emb = apply_rotary_encoding(
                        agg_emb,
                        layer_idx + 1,
                        base_theta=args.himoe_rope_theta
                    )
                else:
                    rotated_emb = agg_emb
                
                # Only measure user embeddings
                stats = compute_embedding_stats(rotated_emb[:args.user])
                layer_stats[f'layer_{layer_idx + 1}'] = stats
                print(f"  Layer {layer_idx + 1}: MAD={stats['mad']:.4f}, AvgCosSim={stats['avg_cosine_sim']:.4f}, "
                      f"Norm={stats['embedding_norm']:.4f}, Std={stats['embedding_std']:.6f}")
                
                current_emb = rotated_emb
        
        results[label] = layer_stats
    
    # Compute improvement
    print(f"\n--- RoPE Effect Summary ---")
    print(f"{'Layer':<10} {'MAD (w/ RoPE)':<15} {'MAD (w/o RoPE)':<16} {'Δ MAD':<10} {'Improvement':<12}")
    print("-"*65)
    
    for l in range(num_layers + 1):
        key = f'layer_{l}'
        mad_with = results['with_rope'][key]['mad']
        mad_without = results['without_rope'][key]['mad']
        delta = mad_with - mad_without
        improvement = delta / max(mad_without, 1e-10) * 100
        print(f"Layer {l:<4} {mad_with:<15.4f} {mad_without:<16.4f} {delta:<10.4f} {improvement:<12.2f}%")
    
    # Plot
    layers = list(range(num_layers + 1))
    mad_with = [results['with_rope'][f'layer_{l}']['mad'] for l in layers]
    mad_without = [results['without_rope'][f'layer_{l}']['mad'] for l in layers]
    cos_with = [results['with_rope'][f'layer_{l}']['avg_cosine_sim'] for l in layers]
    cos_without = [results['without_rope'][f'layer_{l}']['avg_cosine_sim'] for l in layers]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(layers, mad_with, 'o-', color='#2196F3', linewidth=2.5, markersize=8, label='With RoPE')
    ax1.plot(layers, mad_without, 's--', color='#F44336', linewidth=2.5, markersize=8, label='Without RoPE')
    ax1.set_xlabel('Layer', fontsize=16, fontweight='bold')
    ax1.set_ylabel('MAD (Mean Average Distance)', fontsize=16, fontweight='bold')
    ax1.set_title('MAD Across Layers\n(Higher = Less Over-Smoothing)', fontsize=18, fontweight='bold')
    ax1.legend(fontsize=14)
    ax1.tick_params(labelsize=13)
    ax1.set_xticks(layers)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(layers, cos_with, 'o-', color='#2196F3', linewidth=2.5, markersize=8, label='With RoPE')
    ax2.plot(layers, cos_without, 's--', color='#F44336', linewidth=2.5, markersize=8, label='Without RoPE')
    ax2.set_xlabel('Layer', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Avg Cosine Similarity', fontsize=16, fontweight='bold')
    ax2.set_title('Embedding Similarity Across Layers\n(Lower = Less Over-Smoothing)', fontsize=18, fontweight='bold')
    ax2.legend(fontsize=14)
    ax2.tick_params(labelsize=13)
    ax2.set_xticks(layers)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'rope_mad_analysis.pdf')
    plt.savefig(save_path, format='pdf', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {save_path}")
    
    # Save results
    with open(os.path.join(output_dir, 'rope_analysis.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def load_model_from_checkpoint(checkpoint_path, handler):
    """Load model from checkpoint."""
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model = checkpoint['model']
        else:
            model = HiMoE_Model(adj=handler.adj_stu, num_users=args.user, num_items=args.item).cuda()
            model.load_state_dict(checkpoint)
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    for moe_layer in model.moe_layers:
        if not hasattr(moe_layer, 'use_sparse_expert'):
            moe_layer.use_sparse_expert = True
    
    model.eval()
    return model


def main():
    vis_checkpoint = getattr(args, 'vis_checkpoint', None)
    vis_output_dir = getattr(args, 'vis_output_dir', './rebuttal/reviewer2/real_results/rope_analysis')
    
    print("Loading data...")
    handler = DataHandler()
    handler.LoadData()
    
    if not vis_checkpoint:
        raise ValueError("Must specify --vis_checkpoint")
    
    model = load_model_from_checkpoint(vis_checkpoint, handler)
    print(f"Model loaded from {vis_checkpoint}")
    
    results = analyze_rope_effect(model, handler, vis_output_dir)
    
    print(f"\nAll results saved to: {vis_output_dir}")


if __name__ == '__main__':
    main()
