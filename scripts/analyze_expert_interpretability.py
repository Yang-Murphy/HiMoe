"""
Hi-MoE Expert Interpretability Analysis Script
==================================================

Designed for rebuttal to address reviewer concerns:
(a) How expert-specific subgraphs are derived
(b) How routing ensures experts capture different patterns  
(c) Whether experts correspond to interpretable user intents

Experiments:
1. Expert Subgraph Diversity: Jaccard/overlap between expert edge masks
2. Expert Routing vs User Activity: correlation between user activity and routing
3. Expert Intent Clustering: intra vs inter-expert co-interaction analysis
4. Expert Subgraph Topology Analysis: degree distribution, edge retention

Usage:
    python analyze_expert_interpretability.py \
        --config configs/amazon.yaml \
        --vis_checkpoint ./outModels/amazon_high_mem_best.mod \
        --vis_output_dir ./logs/expert_interpretability/amazon
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from tqdm import tqdm
from collections import defaultdict, Counter
from scipy import stats as scipy_stats
from scipy.sparse import csr_matrix, coo_matrix

# 设置全局字体
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif', 'Liberation Serif', 'FreeSerif', 'serif']
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['axes.unicode_minus'] = False

from Params import args
from DataHandler import DataHandler
from HiMoE import HiMoE_Model


class ExpertInterpretabilityAnalyzer:
    """
    Comprehensive analyzer for MoE expert interpretability.
    Addresses reviewer concerns (a), (b), (c).
    """
    
    def __init__(self, model, handler, output_dir='./logs/expert_interpretability'):
        self.model = model
        self.handler = handler
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.model.eval()
        self.num_layers = len(model.moe_layers)
        self.num_experts = model.moe_layers[0].num_experts
        self.top_k = model.moe_layers[0].top_k
        self.num_users = args.user
        self.num_items = args.item
        
        # Load training interaction data
        self.trnMat = self._load_trn_mat()
        
    def _load_trn_mat(self):
        """Load training interaction matrix."""
        trnfile = self.handler.predir + 'trnMat.pkl'
        with open(trnfile, 'rb') as f:
            trnMat = (pickle.load(f) != 0).astype(np.float32)
        if not isinstance(trnMat, csr_matrix):
            trnMat = csr_matrix(trnMat)
        return trnMat
    
    # ==================================================================
    # Experiment 1: Expert Subgraph Diversity (addresses concern a & b)
    # ==================================================================
    
    def analyze_expert_subgraphs(self):
        """
        Analyze expert-specific subgraphs using SOFT (continuous) edge weights.
        Each expert learns a per-edge sigmoid weight in [0,1].
        We use Pearson correlation to measure pairwise diversity.
        Low correlation = experts emphasize different edges = diverse subgraphs.
        """
        print("\n" + "="*70)
        print("Experiment 1: Expert Subgraph Diversity Analysis (Soft Weights)")
        print("="*70)
        
        results = {}
        
        for layer_idx, moe_layer in enumerate(self.model.moe_layers):
            print(f"\n--- Layer {layer_idx} ---")
            
            # Extract soft edge weights
            expert_weights = []  # continuous sigmoid values
            edge_weight_means = []
            
            # Shared expert
            if moe_layer.use_sparse_expert and moe_layer.shared_expert is not None:
                shared_probs = torch.sigmoid(moe_layer.shared_expert.edge_mask_logits).detach().cpu().numpy()
                shared_mean = shared_probs.mean()
                shared_binary_retention = (shared_probs > 0.5).mean()
                print(f"  Shared Expert: mean_weight={shared_mean:.4f}, binary_retention={shared_binary_retention:.4f}")
            
            # Routed experts
            for exp_id, expert in enumerate(moe_layer.routed_experts):
                probs = torch.sigmoid(expert.edge_mask_logits).detach().cpu().numpy()
                expert_weights.append(probs)
                edge_weight_means.append(probs.mean())
            
            expert_weights_arr = np.array(expert_weights)  # [num_experts, num_edges]
            num_exp = len(expert_weights)
            total_edges = expert_weights_arr.shape[1]
            
            print(f"  Soft edge weight statistics:")
            print(f"    Overall range: [{expert_weights_arr.min():.4f}, {expert_weights_arr.max():.4f}]")
            print(f"    Mean per expert: mean={np.mean(edge_weight_means):.4f}, std={np.std(edge_weight_means):.6f}")
            print(f"    Std across experts per edge: {expert_weights_arr.std(axis=0).mean():.6f}")
            
            # Pairwise Pearson correlation (low = diverse)
            from scipy.stats import pearsonr
            corr_matrix = np.eye(num_exp)
            corr_values = []
            
            # For efficiency, sample edges if too many
            if total_edges > 100000:
                sample_idx = np.random.choice(total_edges, 100000, replace=False)
                sampled_weights = expert_weights_arr[:, sample_idx]
            else:
                sampled_weights = expert_weights_arr
            
            for i in range(num_exp):
                for j in range(i+1, num_exp):
                    r, _ = pearsonr(sampled_weights[i], sampled_weights[j])
                    corr_matrix[i, j] = r
                    corr_matrix[j, i] = r
                    corr_values.append(r)
            
            corr_values = np.array(corr_values)
            
            print(f"\n  Pairwise Pearson Correlation (lower = more diverse):")
            print(f"    Mean: {corr_values.mean():.4f}")
            print(f"    Std:  {corr_values.std():.4f}")
            print(f"    Min:  {corr_values.min():.4f}")
            print(f"    Max:  {corr_values.max():.4f}")
            print(f"    → {'Near-zero correlation: experts learn DIVERSE edge patterns' if abs(corr_values.mean()) < 0.1 else 'Experts share similar edge patterns'}")
            
            # Cosine similarity of edge weight vectors
            norms = np.linalg.norm(expert_weights_arr, axis=1, keepdims=True)
            normalized = expert_weights_arr / (norms + 1e-10)
            cosine_matrix = normalized @ normalized.T
            upper_tri_idx = np.triu_indices(num_exp, k=1)
            cosine_values = cosine_matrix[upper_tri_idx]
            
            print(f"\n  Pairwise Cosine Similarity of edge weights:")
            print(f"    Mean: {cosine_values.mean():.4f}")
            print(f"    Std:  {cosine_values.std():.6f}")
            
            # Top-k divergent edges: for each expert, find edges where it has highest relative weight
            # This shows each expert's "preferred" edges
            expert_ranks = np.argsort(-expert_weights_arr, axis=1)  # [num_experts, num_edges], sorted desc
            top_1pct = max(1, total_edges // 100)
            
            # Measure overlap of top-1% edges between expert pairs
            top_overlaps = []
            for i in range(num_exp):
                top_i = set(expert_ranks[i, :top_1pct])
                for j in range(i+1, num_exp):
                    top_j = set(expert_ranks[j, :top_1pct])
                    overlap = len(top_i & top_j) / top_1pct
                    top_overlaps.append(overlap)
            
            top_overlaps = np.array(top_overlaps)
            random_overlap = top_1pct / total_edges  # expected overlap if random
            
            print(f"\n  Top-1% edge overlap between experts:")
            print(f"    Mean overlap: {top_overlaps.mean():.4f}")
            print(f"    Random baseline: {random_overlap:.4f}")
            print(f"    Ratio (actual/random): {top_overlaps.mean() / max(random_overlap, 1e-10):.2f}")
            
            results[f'layer_{layer_idx}'] = {
                'edge_weight_means': [float(x) for x in edge_weight_means],
                'pearson_corr_mean': float(corr_values.mean()),
                'pearson_corr_std': float(corr_values.std()),
                'cosine_sim_mean': float(cosine_values.mean()),
                'cosine_sim_std': float(cosine_values.std()),
                'top1pct_overlap_mean': float(top_overlaps.mean()),
                'top1pct_random_baseline': float(random_overlap),
                'corr_matrix': corr_matrix.tolist(),
            }
            
            # Plot correlation heatmap
            self._plot_correlation_heatmap(corr_matrix, layer_idx)
            
        # Save results
        with open(os.path.join(self.output_dir, 'subgraph_diversity.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def _plot_correlation_heatmap(self, corr_matrix, layer_idx):
        """Plot pairwise Pearson correlation heatmap of expert edge weights."""
        n = corr_matrix.shape[0]
        fig_size = max(8, n * 0.4)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))
        
        annot = n <= 16
        upper_tri_idx = np.triu_indices(n, k=1)
        mean_corr = corr_matrix[upper_tri_idx].mean()
        
        sns.heatmap(
            corr_matrix,
            annot=annot,
            fmt='.2f' if annot else '',
            annot_kws={'size': max(6, 14 - n // 4)} if annot else {},
            cmap='RdBu_r',
            center=0,
            xticklabels=[f'{i}' for i in range(n)],
            yticklabels=[f'{i}' for i in range(n)],
            ax=ax,
            vmin=-0.5, vmax=0.5,
            cbar_kws={'label': 'Pearson Correlation'}
        )
        
        ax.set_title(f'Expert Edge Weight Correlation (Layer {layer_idx})\n'
                     f'Mean r={mean_corr:.4f} (near 0 = diverse)',
                     fontsize=18, fontweight='bold')
        ax.set_xlabel('Expert ID', fontsize=14, fontweight='bold')
        ax.set_ylabel('Expert ID', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f'edge_corr_heatmap_layer{layer_idx}.pdf')
        plt.savefig(save_path, format='pdf', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")
    
    # ==================================================================
    # Experiment 2: Routing vs User Activity (addresses concern b & c)
    # ==================================================================
    
    def analyze_routing_vs_activity(self):
        """
        Analyze if routing is correlated with user activity level.
        - Bin users by interaction count (activity)
        - Check which experts different activity groups prefer
        - Compute routing entropy per activity group
        """
        print("\n" + "="*70)
        print("Experiment 2: Expert Routing vs User Activity Analysis")
        print("="*70)
        
        # Get user activity (number of interactions)
        user_degrees = np.array(self.trnMat.sum(axis=1)).flatten()  # [num_users]
        
        # Get all test users
        test_user_ids = []
        for usr, _ in self.handler.tstLoader:
            test_user_ids.extend(usr.numpy().tolist())
        test_user_ids = sorted(set(test_user_ids))
        
        test_degrees = user_degrees[test_user_ids]
        
        # Bin users into activity groups (quartiles)
        quartiles = np.percentile(test_degrees, [25, 50, 75])
        activity_groups = {}
        group_names = ['Low', 'Medium-Low', 'Medium-High', 'High']
        for i, uid in enumerate(test_user_ids):
            deg = test_degrees[i]
            if deg <= quartiles[0]:
                group = 0
            elif deg <= quartiles[1]:
                group = 1
            elif deg <= quartiles[2]:
                group = 2
            else:
                group = 3
            if group not in activity_groups:
                activity_groups[group] = []
            activity_groups[group].append(uid)
        
        print(f"\nUser activity groups (quartile-based):")
        for g in range(4):
            uids = activity_groups.get(g, [])
            degs = user_degrees[uids] if len(uids) > 0 else []
            print(f"  {group_names[g]}: {len(uids)} users, "
                  f"avg degree={np.mean(degs):.1f}, "
                  f"range=[{np.min(degs):.0f}, {np.max(degs):.0f}]")
        
        # Collect routing decisions per layer
        results = {}
        
        with torch.no_grad():
            if args.use_lrbs:
                embeddings = self.model.lrbs_embedding(apply_mask=True)
            else:
                embeddings = self.model.full_embedding.weight
            
            current_emb = embeddings
            
            for layer_idx, moe_layer in enumerate(self.model.moe_layers):
                print(f"\n--- Layer {layer_idx} ---")
                
                user_emb = current_emb[:self.num_users]
                
                # Get routing probabilities for all test users
                all_test_emb = user_emb[test_user_ids]
                router_logits = moe_layer.router(all_test_emb)
                router_probs = F.softmax(router_logits, dim=-1).cpu().numpy()
                topk_probs, topk_indices = torch.topk(
                    F.softmax(router_logits, dim=-1), self.top_k, dim=-1
                )
                topk_indices = topk_indices.cpu().numpy()
                
                # Analyze per activity group
                group_expert_distributions = {}
                group_entropies = {}
                
                for g in range(4):
                    group_uids = activity_groups.get(g, [])
                    if not group_uids:
                        continue
                    
                    # Find indices in test_user_ids
                    uid_set = set(group_uids)
                    group_indices = [i for i, uid in enumerate(test_user_ids) if uid in uid_set]
                    
                    # Expert selection distribution for this group
                    expert_counts = np.zeros(self.num_experts)
                    for idx in group_indices:
                        for exp_id in topk_indices[idx]:
                            expert_counts[exp_id] += 1
                    
                    total = expert_counts.sum()
                    expert_dist = expert_counts / max(total, 1)
                    
                    # Compute entropy
                    entropy = -np.sum(expert_dist * np.log(expert_dist + 1e-10))
                    max_entropy = np.log(self.num_experts)
                    norm_entropy = entropy / max_entropy
                    
                    group_expert_distributions[g] = expert_dist.tolist()
                    group_entropies[g] = float(norm_entropy)
                    
                    # Top-3 preferred experts
                    top3 = np.argsort(expert_dist)[::-1][:3]
                    print(f"  {group_names[g]} activity: entropy={norm_entropy:.4f}, "
                          f"top experts={list(top3)} "
                          f"({expert_dist[top3[0]]:.3f}, {expert_dist[top3[1]]:.3f}, {expert_dist[top3[2]]:.3f})")
                
                # Test if distributions differ (KL divergence between groups)
                kl_divergences = {}
                for g1 in range(4):
                    for g2 in range(g1+1, 4):
                        if g1 in group_expert_distributions and g2 in group_expert_distributions:
                            p = np.array(group_expert_distributions[g1]) + 1e-10
                            q = np.array(group_expert_distributions[g2]) + 1e-10
                            kl = np.sum(p * np.log(p / q))
                            kl_divergences[f'{group_names[g1]}_vs_{group_names[g2]}'] = float(kl)
                
                print(f"\n  KL divergences between activity groups:")
                for k, v in kl_divergences.items():
                    print(f"    {k}: {v:.4f}")
                
                results[f'layer_{layer_idx}'] = {
                    'group_expert_distributions': {group_names[g]: d for g, d in group_expert_distributions.items()},
                    'group_entropies': {group_names[g]: e for g, e in group_entropies.items()},
                    'kl_divergences': kl_divergences,
                }
                
                # Update embeddings for next layer
                agg_emb, _ = moe_layer(self.handler.adj_stu, current_emb)
                current_emb = agg_emb
        
        # Plot routing distributions per activity group
        self._plot_routing_by_activity(results, group_names)
        
        # Save
        with open(os.path.join(self.output_dir, 'routing_vs_activity.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def _plot_routing_by_activity(self, results, group_names):
        """Plot and compare expert routing distributions across activity groups."""
        for layer_key, layer_data in results.items():
            layer_idx = int(layer_key.split('_')[1])
            dists = layer_data['group_expert_distributions']
            
            fig, ax = plt.subplots(figsize=(max(12, self.num_experts * 0.5), 6))
            
            x = np.arange(self.num_experts)
            width = 0.8 / len(dists)
            colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']
            
            for i, (group_name, dist) in enumerate(dists.items()):
                offset = (i - len(dists)/2 + 0.5) * width
                ax.bar(x + offset, dist, width, label=group_name, color=colors[i], alpha=0.85)
            
            ideal = 1.0 / self.num_experts
            ax.axhline(y=ideal, color='gray', linestyle='--', linewidth=1.5, 
                       label=f'Uniform ({ideal:.3f})', alpha=0.7)
            
            ax.set_xlabel('Expert ID', fontsize=16, fontweight='bold')
            ax.set_ylabel('Selection Probability', fontsize=16, fontweight='bold')
            ax.set_title(f'Expert Routing Distribution by User Activity (Layer {layer_idx})',
                        fontsize=18, fontweight='bold')
            ax.set_xticks(range(0, self.num_experts, max(1, self.num_experts // 16)))
            ax.legend(fontsize=12, title='Activity Level', title_fontsize=13)
            ax.tick_params(axis='both', labelsize=12)
            
            plt.tight_layout()
            save_path = os.path.join(self.output_dir, f'routing_by_activity_layer{layer_idx}.pdf')
            plt.savefig(save_path, format='pdf', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {save_path}")
    
    # ==================================================================
    # Experiment 3: Expert Intent Clustering (addresses concern c)
    # ==================================================================
    
    def analyze_expert_intent_clustering(self):
        """
        Analyze whether experts correspond to interpretable user intents:
        - For each expert, find users predominantly routed to it
        - Compute intra-expert item co-interaction rate
        - Compare with inter-expert co-interaction rate
        - Higher intra/inter ratio → experts capture distinct intents
        """
        print("\n" + "="*70)
        print("Experiment 3: Expert Intent Clustering Analysis")
        print("="*70)
        
        # Get all test users
        test_user_ids = []
        for usr, _ in self.handler.tstLoader:
            test_user_ids.extend(usr.numpy().tolist())
        test_user_ids = sorted(set(test_user_ids))
        
        results = {}
        
        with torch.no_grad():
            if args.use_lrbs:
                embeddings = self.model.lrbs_embedding(apply_mask=True)
            else:
                embeddings = self.model.full_embedding.weight
            
            current_emb = embeddings
            
            for layer_idx, moe_layer in enumerate(self.model.moe_layers):
                print(f"\n--- Layer {layer_idx} ---")
                
                user_emb = current_emb[:self.num_users]
                all_test_emb = user_emb[test_user_ids]
                
                # Get primary expert (top-1) for each user
                router_logits = moe_layer.router(all_test_emb)
                router_probs = F.softmax(router_logits, dim=-1)
                primary_expert = torch.argmax(router_probs, dim=-1).cpu().numpy()
                
                # Group users by primary expert
                expert_user_groups = defaultdict(list)
                for i, uid in enumerate(test_user_ids):
                    expert_user_groups[primary_expert[i]].append(uid)
                
                # Compute user-item interaction sets
                user_items = {}
                for uid in test_user_ids:
                    items = set(self.trnMat[uid].indices.tolist())
                    if len(items) > 0:
                        user_items[uid] = items
                
                # Compute intra-expert item overlap (Jaccard between users in same group)
                intra_overlaps = []
                inter_overlaps = []
                expert_item_stats = {}
                
                active_experts = [e for e in range(self.num_experts) if len(expert_user_groups[e]) >= 2]
                
                print(f"  Active experts (>= 2 users): {len(active_experts)}/{self.num_experts}")
                
                for exp_id in active_experts:
                    group_users = expert_user_groups[exp_id]
                    group_users_with_items = [u for u in group_users if u in user_items]
                    
                    if len(group_users_with_items) < 2:
                        continue
                    
                    # Union of all items in this expert's user group
                    all_items_set = set()
                    for uid in group_users_with_items:
                        all_items_set.update(user_items[uid])
                    
                    expert_item_stats[exp_id] = {
                        'num_users': len(group_users_with_items),
                        'unique_items': len(all_items_set),
                        'avg_items_per_user': np.mean([len(user_items[u]) for u in group_users_with_items])
                    }
                    
                    # Sample pairs for efficiency
                    sample_size = min(500, len(group_users_with_items))
                    sampled = np.random.choice(group_users_with_items, sample_size, replace=False) if sample_size < len(group_users_with_items) else group_users_with_items
                    
                    # Intra-expert Jaccard
                    pair_count = 0
                    for i in range(len(sampled)):
                        for j in range(i+1, min(i+20, len(sampled))):  # limit pairs per user
                            u1, u2 = sampled[i], sampled[j]
                            inter = len(user_items[u1] & user_items[u2])
                            union = len(user_items[u1] | user_items[u2])
                            if union > 0:
                                intra_overlaps.append(inter / union)
                                pair_count += 1
                
                # Inter-expert Jaccard (sample pairs from different expert groups)
                if len(active_experts) >= 2:
                    num_inter_pairs = min(len(intra_overlaps), 5000)
                    pair_count = 0
                    for _ in range(num_inter_pairs):
                        e1, e2 = np.random.choice(active_experts, 2, replace=False)
                        g1 = [u for u in expert_user_groups[e1] if u in user_items]
                        g2 = [u for u in expert_user_groups[e2] if u in user_items]
                        if len(g1) == 0 or len(g2) == 0:
                            continue
                        u1 = np.random.choice(g1)
                        u2 = np.random.choice(g2)
                        inter = len(user_items[u1] & user_items[u2])
                        union = len(user_items[u1] | user_items[u2])
                        if union > 0:
                            inter_overlaps.append(inter / union)
                            pair_count += 1
                
                intra_mean = np.mean(intra_overlaps) if intra_overlaps else 0
                inter_mean = np.mean(inter_overlaps) if inter_overlaps else 0
                ratio = intra_mean / max(inter_mean, 1e-10)
                
                print(f"\n  Item Co-interaction Analysis:")
                print(f"    Intra-expert Jaccard: {intra_mean:.6f} (n={len(intra_overlaps)})")
                print(f"    Inter-expert Jaccard: {inter_mean:.6f} (n={len(inter_overlaps)})")
                print(f"    Intra/Inter ratio:    {ratio:.4f}")
                print(f"    → {'Experts capture DISTINCT intent patterns (ratio > 1)' if ratio > 1.0 else 'Experts share similar patterns'}")
                
                # Statistical significance test
                if len(intra_overlaps) > 30 and len(inter_overlaps) > 30:
                    t_stat, p_value = scipy_stats.mannwhitneyu(
                        intra_overlaps, inter_overlaps, alternative='greater'
                    )
                    print(f"    Mann-Whitney U test: p-value = {p_value:.6e}")
                    print(f"    → {'Statistically significant (p < 0.05)' if p_value < 0.05 else 'Not significant'}")
                else:
                    p_value = None
                
                # Expert group size distribution
                group_sizes = [len(expert_user_groups[e]) for e in range(self.num_experts)]
                print(f"\n  Expert group sizes:")
                print(f"    Mean: {np.mean(group_sizes):.1f}")
                print(f"    Std:  {np.std(group_sizes):.1f}")
                print(f"    Min:  {np.min(group_sizes)}")
                print(f"    Max:  {np.max(group_sizes)}")
                
                results[f'layer_{layer_idx}'] = {
                    'intra_jaccard_mean': float(intra_mean),
                    'inter_jaccard_mean': float(inter_mean),
                    'intra_inter_ratio': float(ratio),
                    'p_value': float(p_value) if p_value is not None else None,
                    'group_sizes': group_sizes,
                    'num_active_experts': len(active_experts),
                    'expert_item_stats': {str(k): v for k, v in expert_item_stats.items()},
                }
                
                # Update embeddings
                agg_emb, _ = moe_layer(self.handler.adj_stu, current_emb)
                current_emb = agg_emb
        
        # Plot
        self._plot_intent_clustering(results)
        
        # Save
        with open(os.path.join(self.output_dir, 'intent_clustering.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def _plot_intent_clustering(self, results):
        """Plot intra vs inter expert co-interaction comparison."""
        layers = sorted(results.keys())
        
        fig, axes = plt.subplots(1, len(layers), figsize=(6 * len(layers), 5))
        if len(layers) == 1:
            axes = [axes]
        
        for ax, layer_key in zip(axes, layers):
            layer_idx = int(layer_key.split('_')[1])
            data = results[layer_key]
            
            categories = ['Intra-Expert', 'Inter-Expert']
            values = [data['intra_jaccard_mean'], data['inter_jaccard_mean']]
            colors = ['#2196F3', '#FF5722']
            
            bars = ax.bar(categories, values, color=colors, width=0.5, alpha=0.85)
            
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.02,
                       f'{val:.5f}', ha='center', va='bottom', fontsize=14, fontweight='bold')
            
            ratio = data['intra_inter_ratio']
            p_val = data.get('p_value')
            p_str = f'\np={p_val:.2e}' if p_val is not None else ''
            ax.set_title(f'Layer {layer_idx}\nRatio={ratio:.2f}{p_str}',
                        fontsize=16, fontweight='bold')
            ax.set_ylabel('Mean Jaccard Similarity', fontsize=14, fontweight='bold')
            ax.tick_params(axis='both', labelsize=12)
        
        plt.suptitle('Intra vs Inter Expert User-Item Co-interaction',
                     fontsize=20, fontweight='bold', y=1.02)
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'intent_clustering.pdf')
        plt.savefig(save_path, format='pdf', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")
    
    # ==================================================================
    # Experiment 4: Embedding Space Separation (addresses concern b)
    # ==================================================================
    
    def analyze_embedding_separation(self):
        """
        Experiment 4: Embedding Space Separation by Expert Assignment.
        Users routed to the same expert should be closer in embedding space.
        Measured via Silhouette score and inter/intra distance ratio.
        """
        print("\n" + "="*70)
        print("Experiment 4: Embedding Space Separation by Expert")
        print("="*70)
        
        test_user_ids = []
        for usr, _ in self.handler.tstLoader:
            test_user_ids.extend(usr.numpy().tolist())
        test_user_ids = sorted(set(test_user_ids))
        
        results = {}
        
        with torch.no_grad():
            if args.use_lrbs:
                embeddings = self.model.lrbs_embedding(apply_mask=True)
            else:
                embeddings = self.model.full_embedding.weight
            
            current_emb = embeddings
            
            for layer_idx, moe_layer in enumerate(self.model.moe_layers):
                print(f"\n--- Layer {layer_idx} ---")
                
                user_emb = current_emb[:self.num_users]
                all_test_emb = user_emb[test_user_ids].cpu().numpy()
                
                router_logits = moe_layer.router(user_emb[test_user_ids])
                router_probs = F.softmax(router_logits, dim=-1)
                primary_expert = torch.argmax(router_probs, dim=-1).cpu().numpy()
                
                expert_groups = defaultdict(list)
                for i, exp_id in enumerate(primary_expert):
                    expert_groups[exp_id].append(i)
                
                active_experts = [e for e in range(self.num_experts) if len(expert_groups[e]) >= 2]
                
                # Compute expert centroids
                centroids = {}
                for exp_id in active_experts:
                    indices = expert_groups[exp_id]
                    centroids[exp_id] = all_test_emb[indices].mean(axis=0)
                
                # Intra-expert distance
                intra_dists = []
                for exp_id in active_experts:
                    indices = expert_groups[exp_id]
                    c = centroids[exp_id]
                    dists = np.linalg.norm(all_test_emb[indices] - c, axis=1)
                    intra_dists.extend(dists.tolist())
                intra_mean = np.mean(intra_dists)
                
                # Inter-expert distance
                centroid_arr = np.array([centroids[e] for e in active_experts])
                inter_dists = []
                for i in range(len(active_experts)):
                    for j in range(i+1, len(active_experts)):
                        d = np.linalg.norm(centroid_arr[i] - centroid_arr[j])
                        inter_dists.append(d)
                inter_mean = np.mean(inter_dists)
                
                ratio = inter_mean / max(intra_mean, 1e-10)
                
                print(f"  Active experts: {len(active_experts)}/{self.num_experts}")
                print(f"  Intra-expert distance (user to centroid): {intra_mean:.4f}")
                print(f"  Inter-expert distance (centroid to centroid): {inter_mean:.4f}")
                print(f"  Inter/Intra ratio: {ratio:.4f} (> 1 = well separated)")
                
                # Silhouette score
                sample_size = min(5000, len(test_user_ids))
                sample_idx = np.random.choice(len(test_user_ids), sample_size, replace=False)
                sample_emb = all_test_emb[sample_idx]
                sample_labels = primary_expert[sample_idx]
                
                unique_labels = np.unique(sample_labels)
                if len(unique_labels) >= 2:
                    from sklearn.metrics import silhouette_score, davies_bouldin_score
                    sil_score = silhouette_score(sample_emb, sample_labels, metric='euclidean',
                                                 sample_size=min(2000, sample_size))
                    db_score = davies_bouldin_score(sample_emb, sample_labels)
                    print(f"  Silhouette score: {sil_score:.4f} (> 0 = meaningful, max 1.0)")
                    print(f"  Davies-Bouldin index: {db_score:.4f} (lower = better separation)")
                else:
                    sil_score = 0.0
                    db_score = float('inf')
                
                # Random baseline
                shuffled_labels = np.random.permutation(sample_labels)
                if len(np.unique(shuffled_labels)) >= 2:
                    sil_random = silhouette_score(sample_emb, shuffled_labels, metric='euclidean',
                                                  sample_size=min(2000, sample_size))
                    print(f"  Random baseline silhouette: {sil_random:.4f}")
                    print(f"  Improvement over random: {sil_score - sil_random:.4f}")
                else:
                    sil_random = 0.0
                
                results[f'layer_{layer_idx}'] = {
                    'intra_distance': float(intra_mean),
                    'inter_distance': float(inter_mean),
                    'inter_intra_ratio': float(ratio),
                    'silhouette_score': float(sil_score),
                    'silhouette_random': float(sil_random),
                    'davies_bouldin': float(db_score),
                    'num_active_experts': len(active_experts),
                }
                
                # Plot PCA visualization
                self._plot_embedding_separation(all_test_emb, primary_expert, active_experts, layer_idx, sil_score)
                
                agg_emb, _ = moe_layer(self.handler.adj_stu, current_emb)
                current_emb = agg_emb
        
        with open(os.path.join(self.output_dir, 'embedding_separation.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def _plot_embedding_separation(self, emb, labels, active_experts, layer_idx, sil_score):
        """Plot PCA of user embeddings colored by expert assignment."""
        from sklearn.decomposition import PCA
        
        n = min(3000, len(emb))
        idx = np.random.choice(len(emb), n, replace=False)
        X = emb[idx]
        L = labels[idx]
        
        pca = PCA(n_components=2)
        X2d = pca.fit_transform(X)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        freq = Counter(L)
        top_experts = [e for e, _ in freq.most_common(8)]
        colors = plt.cm.Set1(np.linspace(0, 1, 8))
        
        other_mask = np.array([l not in top_experts for l in L])
        if other_mask.any():
            ax.scatter(X2d[other_mask, 0], X2d[other_mask, 1], c='lightgray',
                      s=5, alpha=0.3, label='Other experts')
        
        for i, exp_id in enumerate(top_experts):
            mask = L == exp_id
            ax.scatter(X2d[mask, 0], X2d[mask, 1], c=[colors[i]],
                      s=8, alpha=0.5, label=f'Expert {exp_id} (n={mask.sum()})')
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=14, fontweight='bold')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=14, fontweight='bold')
        ax.set_title(f'Layer {layer_idx}: User Embeddings by Expert\n'
                     f'Silhouette={sil_score:.4f}', fontsize=16, fontweight='bold')
        ax.legend(fontsize=9, markerscale=3, ncol=2, loc='best')
        ax.tick_params(labelsize=12)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f'embedding_separation_layer{layer_idx}.pdf')
        plt.savefig(save_path, format='pdf', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")
    
    # ==================================================================
    # Experiment 5: Expert Item Specialization (addresses concern c)
    # ==================================================================
    
    def analyze_expert_item_specialization(self):
        """
        Experiment 5: Expert-Specific Item Specialization.
        Different experts' users interact with different items.
        Measured via Jensen-Shannon divergence vs random baseline.
        """
        print("\n" + "="*70)
        print("Experiment 5: Expert-Specific Item Specialization")
        print("="*70)
        
        test_user_ids = []
        for usr, _ in self.handler.tstLoader:
            test_user_ids.extend(usr.numpy().tolist())
        test_user_ids = sorted(set(test_user_ids))
        
        user_items = {}
        for uid in test_user_ids:
            items = self.trnMat[uid].indices.tolist()
            if items:
                user_items[uid] = items
        
        results = {}
        
        with torch.no_grad():
            if args.use_lrbs:
                embeddings = self.model.lrbs_embedding(apply_mask=True)
            else:
                embeddings = self.model.full_embedding.weight
            
            current_emb = embeddings
            
            for layer_idx, moe_layer in enumerate(self.model.moe_layers):
                print(f"\n--- Layer {layer_idx} ---")
                
                user_emb = current_emb[:self.num_users]
                all_test_emb = user_emb[test_user_ids]
                
                router_logits = moe_layer.router(all_test_emb)
                router_probs = F.softmax(router_logits, dim=-1)
                primary_expert = torch.argmax(router_probs, dim=-1).cpu().numpy()
                
                expert_user_groups = defaultdict(list)
                for i, uid in enumerate(test_user_ids):
                    expert_user_groups[primary_expert[i]].append(uid)
                
                # Per-expert item frequency distribution
                expert_item_dists = {}
                expert_top_items = {}
                active_experts = [e for e in range(self.num_experts)
                                  if len(expert_user_groups[e]) >= 2]
                
                for exp_id in active_experts:
                    item_counter = Counter()
                    for uid in expert_user_groups[exp_id]:
                        if uid in user_items:
                            item_counter.update(user_items[uid])
                    if not item_counter:
                        continue
                    total = sum(item_counter.values())
                    dist = np.zeros(self.num_items)
                    for item_id, count in item_counter.items():
                        if item_id < self.num_items:
                            dist[item_id] = count / total
                    expert_item_dists[exp_id] = dist
                    expert_top_items[exp_id] = [item for item, _ in item_counter.most_common(50)]
                
                # Pairwise JSD
                from scipy.spatial.distance import jensenshannon
                jsd_values = []
                valid_experts = sorted(expert_item_dists.keys())
                
                for i in range(len(valid_experts)):
                    for j in range(i+1, len(valid_experts)):
                        p = expert_item_dists[valid_experts[i]]
                        q = expert_item_dists[valid_experts[j]]
                        jsd = jensenshannon(p, q) ** 2
                        jsd_values.append(jsd)
                
                jsd_values = np.array(jsd_values) if jsd_values else np.array([0])
                
                # Random baseline
                random_jsd_values = []
                for _ in range(3):
                    shuffled = np.random.permutation(primary_expert)
                    rand_groups = defaultdict(list)
                    for i, uid in enumerate(test_user_ids):
                        rand_groups[shuffled[i]].append(uid)
                    rand_dists = {}
                    for exp_id in valid_experts:
                        ic = Counter()
                        for uid in rand_groups.get(exp_id, []):
                            if uid in user_items:
                                ic.update(user_items[uid])
                        if ic:
                            total = sum(ic.values())
                            dist = np.zeros(self.num_items)
                            for item_id, count in ic.items():
                                if item_id < self.num_items:
                                    dist[item_id] = count / total
                            rand_dists[exp_id] = dist
                    for i in range(len(valid_experts)):
                        for j in range(i+1, len(valid_experts)):
                            if valid_experts[i] in rand_dists and valid_experts[j] in rand_dists:
                                jsd = jensenshannon(rand_dists[valid_experts[i]],
                                                   rand_dists[valid_experts[j]]) ** 2
                                random_jsd_values.append(jsd)
                
                random_jsd_values = np.array(random_jsd_values) if random_jsd_values else np.array([0])
                
                # Top-50 item overlap
                top_item_overlaps = []
                for i in range(len(valid_experts)):
                    for j in range(i+1, len(valid_experts)):
                        if valid_experts[i] in expert_top_items and valid_experts[j] in expert_top_items:
                            s1 = set(expert_top_items[valid_experts[i]][:50])
                            s2 = set(expert_top_items[valid_experts[j]][:50])
                            overlap = len(s1 & s2) / 50
                            top_item_overlaps.append(overlap)
                
                top_item_overlaps = np.array(top_item_overlaps) if top_item_overlaps else np.array([0])
                
                improvement = jsd_values.mean() / max(random_jsd_values.mean(), 1e-10)
                
                print(f"  Active experts: {len(valid_experts)}/{self.num_experts}")
                print(f"\n  Inter-expert item distribution divergence (JSD):")
                print(f"    Actual JSD: {jsd_values.mean():.6f}")
                print(f"    Random baseline JSD: {random_jsd_values.mean():.6f}")
                print(f"    Ratio (actual/random): {improvement:.4f}")
                print(f"    -> {'Experts show MORE item specialization than random' if improvement > 1.0 else 'No specialization'}")
                print(f"\n  Top-50 item overlap between expert groups:")
                print(f"    Mean: {top_item_overlaps.mean():.4f} (lower = more specialized)")
                
                results[f'layer_{layer_idx}'] = {
                    'jsd_mean': float(jsd_values.mean()),
                    'jsd_std': float(jsd_values.std()),
                    'random_jsd_mean': float(random_jsd_values.mean()),
                    'jsd_improvement': float(improvement),
                    'top50_overlap_mean': float(top_item_overlaps.mean()),
                    'num_active_experts': len(valid_experts),
                }
                
                # Plot
                self._plot_item_specialization(jsd_values, random_jsd_values, top_item_overlaps, layer_idx)
                
                agg_emb, _ = moe_layer(self.handler.adj_stu, current_emb)
                current_emb = agg_emb
        
        with open(os.path.join(self.output_dir, 'item_specialization.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def _plot_item_specialization(self, jsd_values, random_jsd_values, top_overlaps, layer_idx):
        """Plot item specialization comparison."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.hist(jsd_values, bins=50, color='#2196F3', alpha=0.7,
                label=f'Actual (mean={jsd_values.mean():.5f})', density=True)
        ax1.hist(random_jsd_values, bins=50, color='#FF5722', alpha=0.5,
                label=f'Random (mean={random_jsd_values.mean():.5f})', density=True)
        ax1.set_xlabel('Jensen-Shannon Divergence', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Density', fontsize=14, fontweight='bold')
        ax1.set_title(f'Layer {layer_idx}: Expert Item Distribution Divergence', fontsize=16, fontweight='bold')
        ax1.legend(fontsize=12)
        ax1.tick_params(labelsize=12)
        
        ax2.hist(top_overlaps, bins=30, color='#4CAF50', alpha=0.8, edgecolor='white')
        ax2.axvline(x=top_overlaps.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean={top_overlaps.mean():.3f}')
        ax2.set_xlabel('Top-50 Item Overlap (Jaccard)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('# Expert Pairs', fontsize=14, fontweight='bold')
        ax2.set_title(f'Layer {layer_idx}: Top Item Overlap Between Experts', fontsize=16, fontweight='bold')
        ax2.legend(fontsize=12)
        ax2.tick_params(labelsize=12)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f'item_specialization_layer{layer_idx}.pdf')
        plt.savefig(save_path, format='pdf', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")
    
    # ==================================================================
    # Run all experiments
    # ==================================================================
    
    def run_all(self):
        """Run all interpretability experiments."""
        print("\n" + "█"*70)
        print("█  Hi-MoE Expert Interpretability Analysis")
        print("█  Addressing reviewer concerns (a), (b), (c)")
        print("█"*70)
        
        all_results = {}
        
        # Experiment 1
        all_results['subgraph_diversity'] = self.analyze_expert_subgraphs()
        
        # Experiment 2
        all_results['routing_vs_activity'] = self.analyze_routing_vs_activity()
        
        # Experiment 3
        all_results['intent_clustering'] = self.analyze_expert_intent_clustering()
        
        # Experiment 4
        all_results['embedding_separation'] = self.analyze_embedding_separation()
        
        # Experiment 5
        all_results['item_specialization'] = self.analyze_expert_item_specialization()
        
        # Print final summary
        self._print_final_summary(all_results)
        
        return all_results
    
    def _print_final_summary(self, all_results):
        """Print comprehensive summary for rebuttal."""
        print("\n" + "█"*70)
        print("█  SUMMARY FOR REBUTTAL")
        print("█"*70)
        
        print("\n(a) How expert-specific subgraphs are derived:")
        print("─"*50)
        if 'subgraph_diversity' in all_results:
            for layer_key, data in all_results['subgraph_diversity'].items():
                print(f"  {layer_key}:")
                if 'pearson_corr_mean' in data:
                    print(f"    • Pearson correlation: mean={data['pearson_corr_mean']:.4f} (near 0 = diverse)")
                    print(f"    • Top-1% edge overlap: {data['top1pct_overlap_mean']:.4f} (random={data['top1pct_random_baseline']:.4f})")
                elif 'item_jaccard_mean' in data:
                    print(f"    • Item Jaccard between expert subgraphs: {data['item_jaccard_mean']:.4f}")
                    print(f"    • Degree KS stat: {data['ks_stat_mean']:.4f}")
        
        print("\n(b) How routing ensures experts capture different patterns:")
        print("─"*50)
        if 'embedding_separation' in all_results:
            for layer_key, data in all_results['embedding_separation'].items():
                print(f"  {layer_key}:")
                print(f"    • Silhouette score: {data['silhouette_score']:.4f} (vs random: {data['silhouette_random']:.4f})")
                print(f"    • Inter/Intra distance ratio: {data['inter_intra_ratio']:.4f}")
        if 'routing_vs_activity' in all_results:
            for layer_key, data in all_results['routing_vs_activity'].items():
                kl_vals = list(data.get('kl_divergences', {}).values())
                if kl_vals:
                    print(f"  {layer_key}: KL divergence between activity groups: mean={np.mean(kl_vals):.4f}")
        
        print("\n(c) Whether experts correspond to interpretable user intents:")
        print("─"*50)
        if 'intent_clustering' in all_results:
            for layer_key, data in all_results['intent_clustering'].items():
                print(f"  {layer_key}:")
                print(f"    • Intra-expert Jaccard: {data['intra_jaccard_mean']:.6f}")
                print(f"    • Inter-expert Jaccard: {data['inter_jaccard_mean']:.6f}")
                print(f"    • Ratio (intra/inter): {data['intra_inter_ratio']:.4f}")
                if data.get('p_value') is not None:
                    print(f"    • p-value: {data['p_value']:.2e}")
                print(f"    • Active experts: {data['num_active_experts']}")
        if 'item_specialization' in all_results:
            for layer_key, data in all_results['item_specialization'].items():
                print(f"  {layer_key}:")
                print(f"    • JSD (actual): {data['jsd_mean']:.6f} vs (random): {data['random_jsd_mean']:.6f}")
                print(f"    • Improvement ratio: {data['jsd_improvement']:.4f}x")
                print(f"    • Top-50 item overlap: {data['top50_overlap_mean']:.4f}")
        
        print("\n" + "█"*70)


def load_model_from_checkpoint(checkpoint_path, handler):
    """Load model from checkpoint."""
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model = checkpoint['model']
            print(f"Model loaded from {checkpoint_path} (full model)")
        else:
            model = HiMoE_Model(
                adj=handler.adj_stu,
                num_users=args.user,
                num_items=args.item
            ).cuda()
            model.load_state_dict(checkpoint)
            print(f"Model loaded from {checkpoint_path} (state_dict)")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Compatibility
    for moe_layer in model.moe_layers:
        if not hasattr(moe_layer, 'use_sparse_expert'):
            moe_layer.use_sparse_expert = True
    
    model.eval()
    return model


def main():
    vis_checkpoint = getattr(args, 'vis_checkpoint', None)
    vis_output_dir = getattr(args, 'vis_output_dir', './logs/expert_interpretability')
    
    os.makedirs(vis_output_dir, exist_ok=True)
    
    print("Loading data...")
    handler = DataHandler()
    handler.LoadData()
    
    if vis_checkpoint:
        model = load_model_from_checkpoint(vis_checkpoint, handler)
    else:
        raise ValueError("Must specify --vis_checkpoint for interpretability analysis")
    
    # Check if model has routed experts
    if model.moe_layers[0].num_experts == 0:
        print("Error: Model has no routed experts (num_experts=0)")
        return
    
    # Run analysis
    analyzer = ExpertInterpretabilityAnalyzer(model, handler, output_dir=vis_output_dir)
    analyzer.run_all()
    
    print(f"\nAll results saved to: {vis_output_dir}")


if __name__ == '__main__':
    main()
