# Hi-MoE: Supplementary Experiments and Results

This repository provides supplementary experimental results for the Hi-MoE paper, addressing reviewer concerns during the rebuttal period.

## Architecture Overview

Hi-MoE integrates two components for GNN-based recommendation:
1. **Hybrid Structural MoE**: A shared expert + K routed experts, each operating on a learned subgraph via per-edge sigmoid gating + STE binarization.
2. **Rotary Embedding Propagation (RoPE)**: Layer-depth encoding applied after each aggregation step to preserve geometric distinguishability across propagation depths.

---

## Experiment 1: Expert Intent Clustering

**Goal**: Verify that users routed to the same expert share more similar item preferences.

**Method**: Assign test users to their primary expert (top-1 routing). Compute intra-expert vs inter-expert item Jaccard similarity. Report ratio and Mann-Whitney U p-value.

### Results

| Dataset  | Layer 0 | Layer 1 | Layer 2 | Layer 3 | p-value (L3) |
|----------|---------|---------|---------|---------|---------------|
| Gowalla  | 2.76×   | 3.71×   | 6.34×   | 5.71×   | <10⁻⁷¹        |
| Amazon   | 2.74×   | 3.87×   | 7.22×   | 5.73×   | <10⁻²²        |
| Yelp     | 6.02×   | 5.66×   | 8.49×   | 31.48×  | <10⁻¹⁴        |

**Figures**: See `results/intent_clustering/`

---

## Experiment 2: Routing vs User Activity

**Goal**: Verify that the router differentiates users by behavioral patterns.

**Method**: Partition test users into 4 activity quartiles (Low/Med-Low/Med-High/High by interaction count). Compute KL divergence of routing distributions between groups.

### Results (KL divergence, Low vs High activity)

| Dataset  | Layer 0 | Layer 1 | Layer 2 | Layer 3 |
|----------|---------|---------|---------|---------|
| Gowalla  | 0.376   | 0.397   | 0.540   | 0.755   |
| Amazon   | 0.264   | 0.249   | 0.378   | 0.510   |
| Yelp     | 0.367   | 0.595   | 0.840   | 0.836   |

**Figures**: See `results/routing_vs_activity/`

---

## Experiment 3: Expert Item Specialization

**Goal**: Verify that different experts serve users with different item preferences, compared to random partition.

**Method**: Compute per-expert item frequency distribution. Measure pairwise Jensen-Shannon Divergence (JSD) and compare with random-shuffled baseline. Report ratio and top-50 item overlap.

### Results

| Dataset  | JSD Ratio (L0) | JSD Ratio (L1) | JSD Ratio (L2) | JSD Ratio (L3) | Top-50 Overlap (L3) |
|----------|----------------|----------------|----------------|----------------|---------------------|
| Gowalla  | 1.30×          | 1.37×          | 1.40×          | 1.45×          | 5.5%                |
| Amazon   | 1.29×          | 1.36×          | 1.44×          | 1.50×          | 5.6%                |
| Yelp     | 1.16×          | 1.17×          | 1.16×          | 1.16×          | 3.0%                |

**Figures**: See `results/item_specialization/`

---

## Experiment 4: RoPE Inter-Layer Distinguishability

**Goal**: Verify that RoPE reduces over-smoothing by making layer representations geometrically distinct.

**Method**: Measure average pairwise cosine similarity between all layer embeddings (lower = more distinct = less over-smoothing). Compare with/without RoPE on the same trained model.

### Results

| Dataset  | With RoPE | Without RoPE | Reduction |
|----------|-----------|--------------|-----------|
| Gowalla  | 0.765     | 0.889        | 14.0%     |
| Amazon   | 0.798     | 0.911        | 12.4%     |
| Yelp     | 0.938     | 0.967        | 3.0%      |

**Figures**: See `results/rope_analysis/`

---

## Experiment 5: Cross-Dataset Sparsity Analysis

**Goal**: Explain why Yelp shows K-insensitivity in ablation studies.

**Method**: Analyze user interaction density and routing behavior for sparse (≤3 interactions) vs active (≥10) users.

### Results

| Dataset  | Sparse users (≤3) | Active users (≥10) | Active intent clustering / Sparse |
|----------|-------------------|--------------------|-----------------------------------|
| Gowalla  | 24.3%             | 34.7%              | 3.44×                             |
| Amazon   | 9.8%              | 40.6%              | 2.43×                             |
| Yelp     | 61.6%             | 7.1%               | 1.84×                             |

On Yelp, 61.6% of users have ≤3 interactions, making top-K selection insensitive to K. For the 7.1% active users, routing still produces 1.84× stronger clustering.

---

## Experiment 6: Depth Ablation — Vanilla GCN vs Hi-MoE

**Goal**: Directly compare performance under increasing propagation depth.

**Method**: Train vanilla GCN (no MoE, no RoPE) and Hi-MoE (with RoPE) at L=2,3,4,5,6,8 on Gowalla.

### Results (Gowalla Recall@20)

| Layers | Vanilla GCN | Hi-MoE (w/ RoPE) |
|--------|-------------|-------------------|
| 2      | TBD         | 0.2711            |
| 3      | TBD         | 0.2693            |
| 4      | TBD         | 0.2712            |
| 5      | TBD         | 0.2680            |
| 6      | TBD         | —                 |
| 8      | TBD         | —                 |

*(Vanilla GCN experiments currently running)*

---

## Experiment 7: Shared Expert Contribution

**Goal**: Isolate the contribution of the shared expert (graph denoiser).

**Method**: Train Hi-MoE without the shared expert (routed experts + RoPE only) on Gowalla.

### Results

| Config | Gowalla R@20 |
|--------|-------------|
| Full Hi-MoE | 0.2712 |
| w/o shared expert | TBD |
| w/o routed experts | 0.2675 |

*(Running)*

---

## Configuration

All experiments use:
- **Gowalla**: 25,557 users, 19,747 items, 294,983 interactions
- **Amazon**: 76,469 users, 83,761 items, 966,680 interactions  
- **Yelp**: 42,712 users, 26,822 items, 182,357 interactions

Default Hi-MoE config: embedding_dim=1024, num_experts=32, top_k=4, num_layers=4, RoPE theta=10000.

## Reproducibility

All experiment scripts and analysis code are included in the `scripts/` directory.
