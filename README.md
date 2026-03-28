# Hi-MoE: Supplementary Experiments

## Exp 1: Expert Intent Clustering (Intra/Inter Jaccard Ratio)

|Dataset|L0|L1|L2|L3|p-value|
|--|--|--|--|--|--|
|Gowalla|2.76×|3.71×|6.34×|5.71×|<10⁻⁷¹|
|Amazon|2.74×|3.87×|7.22×|5.73×|<10⁻²²|
|Yelp|6.02×|5.66×|8.49×|31.48×|<10⁻¹⁴|

|Gowalla|Amazon|Yelp|
|--|--|--|
|![](images/intent_clustering_gowalla.png)|![](images/intent_clustering_amazon.png)|![](images/intent_clustering_yelp.png)|

## Exp 2: Routing vs User Activity (KL Divergence, Low vs High)

|Dataset|L0|L1|L2|L3|
|--|--|--|--|--|
|Gowalla|0.376|0.397|0.540|0.755|
|Amazon|0.264|0.249|0.378|0.510|
|Yelp|0.367|0.595|0.840|0.836|

|Gowalla|Amazon|Yelp|
|--|--|--|
|![](images/routing_activity_gowalla_L3.png)|![](images/routing_activity_amazon_L3.png)|![](images/routing_activity_yelp_L3.png)|

## Exp 3: Expert Item Specialization (JSD / Random)

|Dataset|L0|L1|L2|L3|Top-50 Overlap|
|--|--|--|--|--|--|
|Gowalla|1.30×|1.37×|1.40×|1.45×|5.5%|
|Amazon|1.29×|1.36×|1.44×|1.50×|5.6%|
|Yelp|1.16×|1.17×|1.16×|1.16×|3.0%|

|Gowalla|Amazon|Yelp|
|--|--|--|
|![](images/item_spec_gowalla_L3.png)|![](images/item_spec_amazon_L3.png)|![](images/item_spec_yelp_L3.png)|

## Exp 4: RoPE Inter-Layer Cosine Similarity (Cross-Dataset)

|Dataset|With RoPE|Without RoPE|Reduction|
|--|--|--|--|
|Gowalla|0.765|0.889|14.0%|
|Amazon|0.798|0.911|12.4%|
|Yelp|0.938|0.967|3.0%|

|Gowalla|Amazon|Yelp|
|--|--|--|
|![](images/rope_distinguish_gowalla.png)|![](images/rope_distinguish_amazon.png)|![](images/rope_distinguish_yelp.png)|

## Exp 5: RoPE Depth Ablation — LightGCN vs Hi-MoE (Gowalla, Inter-layer Cosine Similarity)

|Layers|LightGCN|Hi-MoE|Reduction|
|--|--|--|--|
|2|0.887|0.882|0.5%|
|3|0.848|0.791|6.7%|
|4|0.808|0.722|10.7%|
|5|0.788|0.623|21.0%|
|6|0.771|0.582|24.5%|
|8|0.736|0.507|31.1%|

## Exp 6: Cross-Dataset Sparsity Analysis

|Dataset|Sparse (≤3)|Active (≥10)|Active/Sparse Clustering|
|--|--|--|--|
|Gowalla|24.3%|34.7%|3.44×|
|Amazon|9.8%|40.6%|2.43×|
|Yelp|61.6%|7.1%|1.84×|

## Exp 7: Expert Routing by User Degree (Gowalla)

|Group|Top-3 Experts (L0)|Top-3 Experts (L1)|
|--|--|--|
|Cold(≤3)|E22,E16,E18|E30,E5,E3|
|Active(≥16)|E3,E2,E30|E19,E12,E31|

## Exp 8: Expert Population Analysis (Gowalla L1)

|Expert|#Users|AvgDegree|Role|
|--|--|--|--|
|E5|1630|6.8|Cold-start|
|E4|1164|19.8|Active|
|E3|667|7.1|Cold-start|
|E2|580|8.0|Low-activity|

## Exp 9: Training Efficiency

|Dataset|Model|Time/Epoch(s)|PeakGPU(MB)|
|--|--|--|--|
|Gowalla|LightGCN|0.88|2888|
|Gowalla|LightGNN|8.05|2120|
|Gowalla|Hi-MoE|4.28|8861|
|Yelp|LightGCN|0.49|3424|
|Yelp|Hi-MoE|2.64|9976|

## Exp 10: Standard Deviations (5 Seeds)

|Dataset|Recall@20|NDCG@20|
|--|--|--|
|Gowalla|0.2598±0.0015|0.1699±0.0010|
|Yelp|0.1346±0.0010|0.0559±0.0004|
|Amazon|0.1479±0.0016|0.1057±0.0016|

All PDF figures available in `results/`.
