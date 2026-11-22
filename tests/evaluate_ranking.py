"""
evaluate_ranking.py
===================
Offline leave-one-out ranking evaluation for hybrid recommender weight configurations.
Includes:
1. Signal Ablation Analysis (CF + each signal added one-at-a-time).
2. Easy / Uniform Negatives vs. Hard Negatives (same subject area & similar difficulty).
3. 95% Bootstrap Confidence Intervals via PyTorch GPU acceleration.
Directly invokes HybridRecommender._hybrid_scores_batch() for 100% production scoring fidelity.
"""

import sys
import os
import time
import json
import numpy as np
import pandas as pd
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)
from hybrid_recommender import HybridRecommender

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] Running evaluation on: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

print("[Platform] Loading HybridRecommender...")
rec = HybridRecommender(retrain=False)
users_df = pd.read_csv("data/users.csv")
interactions_df = pd.read_csv("data/interactions.csv")
courses_df = rec.courses_df

# Filter eligible users: at least 2 completed interactions
completed_counts = interactions_df[interactions_df["completed"] == 1].groupby("user_id").size()
eligible_uids = completed_counts[completed_counts >= 2].index.tolist()
print(f"[Dataset] Total users: {len(users_df)}, Eligible users (>=2 completed): {len(eligible_uids)}")

# Sample 500 users deterministically
rng = np.random.default_rng(42)
n_eval = 500
eval_uids = rng.choice(eligible_uids, size=n_eval, replace=False)
print(f"[Sample] Selected {len(eval_uids)} users for leave-one-out ranking benchmark.\n")

cid_to_course_idx = {str(cid): i for i, cid in enumerate(courses_df["course_id"].astype(str))}

def build_evaluation_pools(neg_type="uniform"):
    pools = []
    for uid in eval_uids:
        u_iacts = interactions_df[interactions_df["user_id"] == uid]
        held_cid = str(u_iacts[u_iacts["completed"] == 1].iloc[-1]["course_id"])
        held_row = courses_df[courses_df["course_id"] == held_cid].iloc[0]
        train_seen = set(u_iacts["course_id"].astype(str)) - {held_cid}
        
        if neg_type == "uniform":
            unseen = courses_df[~courses_df["course_id"].isin(train_seen) & (courses_df["course_id"] != held_cid)]
            sample_neg = unseen.sample(99, random_state=42)
        elif neg_type == "hard":
            same_subj = courses_df[(courses_df["subject"] == held_row["subject"]) & 
                                   (~courses_df["course_id"].isin(train_seen)) & 
                                   (courses_df["course_id"] != held_cid)]
            same_diff = same_subj[same_subj["difficulty_label"] == held_row["difficulty_label"]]
            if len(same_diff) >= 99:
                sample_neg = same_diff.sample(99, random_state=42)
            else:
                needed = 99 - len(same_diff)
                diff_remainder = same_subj[same_subj["difficulty_label"] != held_row["difficulty_label"]]
                if len(diff_remainder) >= needed:
                    sample_neg = pd.concat([same_diff, diff_remainder.sample(needed, random_state=42)])
                else:
                    unseen_gen = courses_df[~courses_df["course_id"].isin(train_seen) & (courses_df["course_id"] != held_cid)]
                    sample_neg = pd.concat([same_subj, unseen_gen.sample(99 - len(same_subj), random_state=42)])
                    
        hr = held_row.to_frame().T
        candidates = pd.concat([sample_neg, hr], ignore_index=True)
        
        query = str(held_row["subject"])
        expanded = rec.embed_engine._expand(query)
        qv = rec.embed_engine.encoder.transform([expanded])[0].reshape(1, -1)
        c_indices = [cid_to_course_idx[str(cid)] for cid in candidates["course_id"]]
        sims = np.dot(rec.embed_engine.embeddings[c_indices], qv.T).flatten()
        candidates["_sem_sim"] = np.clip(sims, 0.0, 1.0)
        candidates["_bm25_raw"] = rec.embed_engine.bm25.score(query)[c_indices]
        
        profile = rec._get_profile(uid)
        scored_base = rec._hybrid_scores_batch(
            candidates, profile, 
            {"semantic": 0.20, "keyword": 0.20, "cf": 0.20, "adaptive": 0.20, "pop": 0.20}
        )
        
        pools.append({
            "uid": uid,
            "profile": profile,
            "candidates": candidates,
            "held_cid": held_cid,
            "scored_base": scored_base
        })
    return pools

print("[Pools] Generating uniform (easy) candidate pools...")
t0 = time.time()
uniform_pools = build_evaluation_pools("uniform")
print(f"[Pools] Uniform pools generated in {time.time()-t0:.1f}s.")

print("[Pools] Generating hard negative candidate pools (same subject & similar difficulty)...")
t1 = time.time()
hard_pools = build_evaluation_pools("hard")
print(f"[Pools] Hard negative pools generated in {time.time()-t1:.1f}s.")

configs_to_test = [
    ("CF Only (Baseline)",        {"semantic": 0.00, "keyword": 0.00, "cf": 1.00, "adaptive": 0.00, "pop": 0.00}),
    ("CF + Semantic (70/30)",     {"semantic": 0.30, "keyword": 0.00, "cf": 0.70, "adaptive": 0.00, "pop": 0.00}),
    ("CF + Semantic (50/50)",     {"semantic": 0.50, "keyword": 0.00, "cf": 0.50, "adaptive": 0.00, "pop": 0.00}),
    ("CF + Keyword (70/30)",      {"semantic": 0.00, "keyword": 0.30, "cf": 0.70, "adaptive": 0.00, "pop": 0.00}),
    ("CF + Keyword (50/50)",      {"semantic": 0.00, "keyword": 0.50, "cf": 0.50, "adaptive": 0.00, "pop": 0.00}),
    ("CF + Adaptive (70/30)",     {"semantic": 0.00, "keyword": 0.00, "cf": 0.70, "adaptive": 0.30, "pop": 0.00}),
    ("CF + Adaptive (50/50)",     {"semantic": 0.00, "keyword": 0.00, "cf": 0.50, "adaptive": 0.50, "pop": 0.00}),
    ("CF + Popularity (70/30)",   {"semantic": 0.00, "keyword": 0.00, "cf": 0.70, "adaptive": 0.00, "pop": 0.30}),
    ("CF + Popularity (50/50)",   {"semantic": 0.00, "keyword": 0.00, "cf": 0.50, "adaptive": 0.00, "pop": 0.50}),
    ("Full Hybrid (Config B)",    {"semantic": 0.20, "keyword": 0.15, "cf": 0.45, "adaptive": 0.15, "pop": 0.05}),
    ("Current Default Hybrid",   {"semantic": 0.30, "keyword": 0.25, "cf": 0.25, "adaptive": 0.15, "pop": 0.05}),
]

def evaluate_pool(pools, label="UNIFORM NEGATIVES"):
    print("\n" + "=" * 115)
    print(f"  EVALUATION RESULTS: {label} (500 USERS, 100 CANDIDATES EACH)")
    print("=" * 115)
    print(f"{'Configuration':<28} {'Weights (Sem/Kw/CF/Ad/Pop)':<28} {'Recall@10':<16} {'NDCG@10 (Mean ± Std)':<24} {'95% Bootstrap CI':<18}")
    print("-" * 115)
    
    k = 10
    n_bootstrap = 1000
    res_list = []
    
    for cfg_name, w in configs_to_test:
        recalls, ndcgs = [], []
        for u in pools:
            if cfg_name == "CF Only (Baseline)":
                scores = u["scored_base"]["cf_score"].values
            else:
                sb = u["scored_base"]
                scores = (w["semantic"] * sb["semantic_score"].values +
                          w["keyword"] * sb["keyword_score"].values +
                          w["cf"] * sb["cf_score"].values +
                          w["adaptive"] * sb["adaptive_score"].values +
                          w["pop"] * sb["popularity_score"].values)
            
            top_k_indices = np.argsort(scores)[::-1][:k]
            top_k_cids = u["candidates"]["course_id"].values[top_k_indices]
            target = u["held_cid"]
            
            if target in top_k_cids:
                rank = int(np.where(top_k_cids == target)[0][0]) + 1
                recalls.append(1.0)
                ndcgs.append(1.0 / np.log2(rank + 1))
            else:
                recalls.append(0.0)
                ndcgs.append(0.0)
                
        r_arr = np.array(recalls)
        n_arr = np.array(ndcgs)
        
        n_tensor = torch.tensor(n_arr, dtype=torch.float32, device=device)
        bs_idx = torch.randint(0, len(n_tensor), (n_bootstrap, len(n_tensor)), device=device)
        bs_means = n_tensor[bs_idx].mean(dim=1).cpu().numpy()
        ci_lo, ci_hi = float(np.percentile(bs_means, 2.5)), float(np.percentile(bs_means, 97.5))
        
        m_rec, s_rec = float(r_arr.mean()), float(r_arr.std())
        m_ndcg, s_ndcg = float(n_arr.mean()), float(n_arr.std())
        
        w_str = f"{w['semantic']:.2f}/{w['keyword']:.2f}/{w['cf']:.2f}/{w['adaptive']:.2f}/{w['pop']:.2f}"
        rec_str = f"{m_rec:.4f} ± {s_rec:.3f}"
        ndcg_str = f"{m_ndcg:.4f} ± {s_ndcg:.4f}"
        ci_str = f"[{ci_lo:.4f}, {ci_hi:.4f}]"
        
        print(f"{cfg_name:<28} {w_str:<28} {rec_str:<16} {ndcg_str:<24} {ci_str:<18}")
        
        res_list.append({
            "config_name": cfg_name,
            "weights": w,
            "recall_at_10": round(m_rec, 4),
            "recall_std": round(s_rec, 4),
            "ndcg_at_10": round(m_ndcg, 4),
            "ndcg_std": round(s_ndcg, 4),
            "ndcg_ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
        })
    print("=" * 115)
    return res_list

uniform_results = evaluate_pool(uniform_pools, "UNIFORM / EASY NEGATIVES")
hard_results = evaluate_pool(hard_pools, "HARD NEGATIVES (SAME SUBJECT & SIMILAR DIFFICULTY)")

# Save to data/model_metrics.json
metrics_path = "data/model_metrics.json"
try:
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
except Exception:
    metrics = {}

metrics["ranking_eval_uniform"] = uniform_results
metrics["ranking_eval_hard_negatives"] = hard_results
# Retain ranking_eval pointing to hard negatives for primary honest tracking
metrics["ranking_eval"] = hard_results
metrics.pop("ranking_metrics", None)

with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\n[Metrics] Successfully updated {metrics_path} with 'ranking_eval_uniform' and 'ranking_eval_hard_negatives'.")
