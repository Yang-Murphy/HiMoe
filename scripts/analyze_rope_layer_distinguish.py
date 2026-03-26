"""
RoPE Layer-wise Distinguishability Analysis
============================================

RoPE's anti-over-smoothing mechanism works by rotating embeddings at each layer
by different angles, making layer-wise representations geometrically distinct.
This means when layers are aggregated (summed), each layer contributes
NON-REDUNDANT information.

Metrics:
1. Inter-layer cosine similarity: Lower = layers are more distinct
2. Effective rank of stacked layer embeddings: Higher = less redundancy
3. Layer contribution orthogonality: How independent each layer's contribution is

Usage:
    python analyze_rope_layer_distinguish.py \
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

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif', 'Liberation Serif', 'FreeSerif', 'serif']
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['axes.unicode_minus'] = False

from Params import args
from DataHandler import DataHandler
from HiMoE import HiMoE_Model, apply_rotary_encoding


def compute_effective_rank(embeddings, sample_size=5000):
    """
    Effective rank via Shannon entropy of normalized singular values.
    Higher = more dimensions carry meaningful information = less redundancy.
    """
    N = embeddings.shape[0]
    if N > sample_size:
        idx = np.random.choice(N, sample_size, replace=False)
        emb = embeddings[idx]
    else:
        emb = embeddings
    
    # Center
    emb = emb - emb.mean(dim=0, keepdim=True)
    
    # SVD
    U, S, V = torch.svd(emb)
    
    # Normalized singular values as probability distribution
    S_norm = S / S.sum()
    S_norm = S_norm[S_norm > 1e-10]  # remove zeros
    
    # Shannon entropy
    entropy = -(S_norm * torch.log(S_norm)).sum().item()
    
    # Effective rank = exp(entropy)
    eff_rank = np.exp(entropy)
    
    # Also compute fraction of variance explained by top-k
    total_var = (S ** 2).sum().item()
    top10_var = (S[:10] ** 2).sum().item() / total_var
    top50_var = (S[:50] ** 2).sum().item() / total_var
    
    return {
        'effective_rank': eff_rank,
        'entropy': entropy,
        'top10_variance_ratio': top10_var,
        'top50_variance_ratio': top50_var,
        'max_rank': len(S_norm),
    }


def analyze_layer_distinguishability(model, handler, output_dir):
    """
    Core analysis: compare layer-wise embedding properties with vs without RoPE.
    """
    print("\n" + "="*70)
    print("RoPE Layer Distinguishability Analysis")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    
    num_layers = len(model.moe_layers)
    adj = handler.adj_stu
    sample_size = 5000
    
    results = {}
    
    for use_rope, label in [(True, 'with_rope'), (False, 'without_rope')]:
        print(f"\n{'='*50}")
        print(f"  {'With' if use_rope else 'Without'} RoPE")
        print(f"{'='*50}")
        
        with torch.no_grad():
            if args.use_lrbs:
                embeddings_0 = model.lrbs_embedding(apply_mask=True)
            else:
                embeddings_0 = model.full_embedding.weight
            
            layer_embeddings = [embeddings_0[:args.user]]  # only users
            current_emb = embeddings_0
            
            for layer_idx, moe_layer in enumerate(model.moe_layers):
                agg_emb, _ = moe_layer(adj, current_emb)
                
                if use_rope:
                    out_emb = apply_rotary_encoding(agg_emb, layer_idx + 1, base_theta=args.himoe_rope_theta)
                else:
                    out_emb = agg_emb
                
                layer_embeddings.append(out_emb[:args.user])
                current_emb = out_emb
        
        # === Metric 1: Inter-layer cosine similarity ===
        print(f"\n  [Metric 1] Inter-layer Cosine Similarity (lower = more distinct)")
        
        # Sample users
        n_users = layer_embeddings[0].shape[0]
        idx = np.random.choice(n_users, min(sample_size, n_users), replace=False)
        
        num_total_layers = len(layer_embeddings)
        inter_layer_sim = np.zeros((num_total_layers, num_total_layers))
        
        for i in range(num_total_layers):
            for j in range(num_total_layers):
                # Per-user cosine similarity between layer i and layer j
                ei = F.normalize(layer_embeddings[i][idx], dim=1)
                ej = F.normalize(layer_embeddings[j][idx], dim=1)
                sim = (ei * ej).sum(dim=1).mean().item()
                inter_layer_sim[i, j] = sim
        
        # Adjacent layer similarity
        adj_sims = [inter_layer_sim[i, i+1] for i in range(num_total_layers - 1)]
        avg_adj_sim = np.mean(adj_sims)
        
        print(f"    Adjacent layer similarities: {[f'{s:.4f}' for s in adj_sims]}")
        print(f"    Average adjacent layer sim: {avg_adj_sim:.4f}")
        
        # Average off-diagonal similarity
        mask = ~np.eye(num_total_layers, dtype=bool)
        avg_offdiag_sim = inter_layer_sim[mask].mean()
        print(f"    Average all-pair sim: {avg_offdiag_sim:.4f}")
        
        # === Metric 2: Effective rank of stacked layer embeddings ===
        print(f"\n  [Metric 2] Effective Rank of Stacked Layers")
        
        # Stack all layer embeddings: [N, D * num_layers]
        stacked = torch.cat([le[idx] for le in layer_embeddings], dim=1)  # [n, D*(L+1)]
        rank_stats = compute_effective_rank(stacked)
        
        print(f"    Effective rank: {rank_stats['effective_rank']:.2f} / {rank_stats['max_rank']}")
        print(f"    Top-10 singular values explain: {rank_stats['top10_variance_ratio']*100:.1f}% variance")
        print(f"    Top-50 singular values explain: {rank_stats['top50_variance_ratio']*100:.1f}% variance")
        
        # === Metric 3: Per-layer effective rank ===
        print(f"\n  [Metric 3] Per-layer Effective Rank")
        per_layer_ranks = []
        for l in range(num_total_layers):
            r = compute_effective_rank(layer_embeddings[l][idx])
            per_layer_ranks.append(r['effective_rank'])
            print(f"    Layer {l}: effective rank = {r['effective_rank']:.2f}")
        
        # === Metric 4: Layer contribution independence ===
        print(f"\n  [Metric 4] Layer Contribution Independence")
        
        # For the final sum embedding, how much does each layer contribute?
        final_sum = sum(layer_embeddings)
        final_sum_sampled = final_sum[idx]
        
        # Project final embedding onto each layer's subspace
        # cos(final, layer_i) measures how aligned the final result is with layer i
        layer_contributions = []
        for l in range(num_total_layers):
            le = layer_embeddings[l][idx]
            cos_sim = F.cosine_similarity(final_sum_sampled, le, dim=1).mean().item()
            norm_ratio = le.norm(dim=1).mean().item() / final_sum_sampled.norm(dim=1).mean().item()
            layer_contributions.append({
                'cosine_to_final': cos_sim,
                'norm_ratio': norm_ratio,
            })
            print(f"    Layer {l}: cos(sum, layer)={cos_sim:.4f}, norm_ratio={norm_ratio:.4f}")
        
        # Contribution variance (lower = more balanced)
        cos_values = [lc['cosine_to_final'] for lc in layer_contributions]
        contribution_std = np.std(cos_values)
        print(f"    Contribution std: {contribution_std:.4f} (lower = more balanced)")
        
        results[label] = {
            'inter_layer_sim': inter_layer_sim.tolist(),
            'avg_adjacent_sim': float(avg_adj_sim),
            'avg_offdiag_sim': float(avg_offdiag_sim),
            'stacked_effective_rank': float(rank_stats['effective_rank']),
            'stacked_top10_var': float(rank_stats['top10_variance_ratio']),
            'stacked_top50_var': float(rank_stats['top50_variance_ratio']),
            'per_layer_ranks': per_layer_ranks,
            'layer_contributions': layer_contributions,
            'contribution_std': float(contribution_std),
        }
    
    # === Print comparison ===
    print(f"\n{'='*70}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*70}")
    
    wr = results['with_rope']
    wor = results['without_rope']
    
    print(f"\n{'Metric':<35} {'With RoPE':<15} {'Without RoPE':<15} {'Δ':<10}")
    print("-"*75)
    print(f"{'Avg adjacent layer similarity':<35} {wr['avg_adjacent_sim']:<15.4f} {wor['avg_adjacent_sim']:<15.4f} {wr['avg_adjacent_sim']-wor['avg_adjacent_sim']:<10.4f}")
    print(f"{'Avg all-pair layer similarity':<35} {wr['avg_offdiag_sim']:<15.4f} {wor['avg_offdiag_sim']:<15.4f} {wr['avg_offdiag_sim']-wor['avg_offdiag_sim']:<10.4f}")
    print(f"{'Stacked effective rank':<35} {wr['stacked_effective_rank']:<15.2f} {wor['stacked_effective_rank']:<15.2f} {wr['stacked_effective_rank']-wor['stacked_effective_rank']:<10.2f}")
    print(f"{'Top-10 SV variance ratio':<35} {wr['stacked_top10_var']*100:<14.1f}% {wor['stacked_top10_var']*100:<14.1f}% {(wr['stacked_top10_var']-wor['stacked_top10_var'])*100:<10.1f}%")
    print(f"{'Contribution balance (std)':<35} {wr['contribution_std']:<15.4f} {wor['contribution_std']:<15.4f} {wr['contribution_std']-wor['contribution_std']:<10.4f}")
    
    # === Plot ===
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    
    # (a) Inter-layer similarity heatmap comparison
    for ax_idx, (lbl, data_key) in enumerate([(r'With RoPE', 'with_rope'), (r'Without RoPE', 'without_rope')]):
        if ax_idx >= 2:
            break
        ax = axes[ax_idx]
        sim = np.array(results[data_key]['inter_layer_sim'])
        import seaborn as sns
        sns.heatmap(sim, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=ax,
                   xticklabels=[f'L{i}' for i in range(sim.shape[0])],
                   yticklabels=[f'L{i}' for i in range(sim.shape[0])],
                   vmin=0, vmax=1, annot_kws={'size': 12, 'weight': 'bold'},
                   cbar_kws={'label': 'Cosine Similarity'})
        ax.set_title(f'{lbl}\nAvg off-diag: {results[data_key]["avg_offdiag_sim"]:.4f}',
                    fontsize=16, fontweight='bold')
        ax.tick_params(labelsize=12)
    
    # (c) Per-layer effective rank comparison
    ax = axes[2]
    layers = list(range(num_layers + 1))
    ax.plot(layers, wr['per_layer_ranks'], 'o-', color='#2196F3', linewidth=2.5, markersize=8, label='With RoPE')
    ax.plot(layers, wor['per_layer_ranks'], 's--', color='#F44336', linewidth=2.5, markersize=8, label='Without RoPE')
    ax.set_xlabel('Layer', fontsize=16, fontweight='bold')
    ax.set_ylabel('Effective Rank', fontsize=16, fontweight='bold')
    ax.set_title(f'Per-Layer Effective Rank\nStacked: {wr["stacked_effective_rank"]:.1f} vs {wor["stacked_effective_rank"]:.1f}',
                fontsize=16, fontweight='bold')
    ax.legend(fontsize=13)
    ax.tick_params(labelsize=13)
    ax.set_xticks(layers)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'rope_layer_distinguish.pdf')
    plt.savefig(save_path, format='pdf', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {save_path}")
    
    with open(os.path.join(output_dir, 'rope_layer_distinguish.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def load_model_from_checkpoint(checkpoint_path, handler):
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
    
    analyze_layer_distinguishability(model, handler, vis_output_dir)
    print(f"\nDone! Results in: {vis_output_dir}")


if __name__ == '__main__':
    main()
