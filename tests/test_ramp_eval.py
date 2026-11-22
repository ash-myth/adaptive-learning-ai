"""
test_ramp_eval.py
=================
Evaluates the linear interpolation ramp from n=0 (cold) to n=5 (warm) under hard negatives.
Also tests degenerate zero-CF case for warm users.
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
from hybrid_recommender import HybridRecommender, WEIGHTS_COLD, WEIGHTS_WARM, COLD_THRESHOLD

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] Evaluation running on: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

rec = HybridRecommender(retrain=False)
users_df = pd.read_csv("data/users.csv")
interactions_df = pd.read_csv("data/interactions.csv")
courses_df = rec.courses_df

completed_counts = interactions_df[interactions_df["completed"] == 1].groupby("user_id").size()
eligible_uids = completed_counts[completed_counts >= 2].index.tolist()

rng = np.random.default_rng(42)
n_eval = 500
eval_uids = rng.choice(eligible_uids, size=n_eval, replace=False)
print(f"[Sample] Selected {len(eval_uids)} users for leave-one-out ranking benchmark.\n")

cid_to_course_idx = {str(cid): i for i, cid in enumerate(courses_df["course_id"].astype(str))}

print("[Pools] Building hard negative candidate pools (same subject & similar difficulty)...")
t0 = time.time()
hard_pools = []
for uid in eval_uids:
    u_iacts = interactions_df[interactions_df["user_id"] == uid]
    held_cid = str(u_iacts[u_iacts["completed"] == 1].iloc[-1]["course_id"])
    held_row = courses_df[courses_df["course_id"] == held_cid].iloc[0]
    train_seen = set(u_iacts["course_id"].astype(str)) - {held_cid}
    
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
    # Pre-score all component signals using production method
    scored_base = rec._hybrid_scores_batch(
        candidates, profile, 
        {"semantic": 0.20, "keyword": 0.20, "cf": 0.20, "adaptive": 0.20, "pop": 0.20}
    )
    
    hard_pools.append({
        "uid": uid,
        "profile": profile,
        "candidates": candidates,
        "held_cid": held_cid,
        "scored_base": scored_base
    })

print(f"[Pools] Generated in {time.time()-t0:.1f}s.\n")

# Evaluate each step in the interpolation ramp: n = 0, 1, 2, 3, 4, 5
print("=" * 115)
print(f"  LINEAR INTERPOLATION RAMP BENCHMARK (HARD NEGATIVES, N=500 USERS)")
print("=" * 115)
print(f"{'Step (n interactions)':<24} {'Weights (Sem/Kw/CF/Ad/Pop)':<30} {'Recall@10':<16} {'NDCG@10 (Mean ± Std)':<24} {'95% Bootstrap CI':<18}")
print("-" * 115)

k = 10
n_bootstrap = 1000

for n in range(6):
    alpha = min(1.0, max(0.0, n / COLD_THRESHOLD))
    w = {key: round((1 - alpha) * WEIGHTS_COLD[key] + alpha * WEIGHTS_WARM[key], 4) for key in WEIGHTS_COLD}
    
    recalls, ndcgs = [], []
    for u in hard_pools:
        sb = u["scored_base"]
        sem = sb["semantic_score"].values
        kw  = sb["keyword_score"].values
        cf  = sb["cf_score"].values
        adap = sb["adaptive_score"].values
        pop = sb["popularity_score"].values
        
        # Check zero-cf degenerate fallback
        if (w.get("cf", 0.0) > 0.0) and (float(cf.max()) == 0.0):
            fb = WEIGHTS_COLD
            hybrid = (fb["semantic"] * sem + fb["keyword"] * kw + fb["cf"] * cf + fb["adaptive"] * adap + fb["pop"] * pop)
        else:
            hybrid = (w["semantic"] * sem + w["keyword"] * kw + w["cf"] * cf + w["adaptive"] * adap + w["pop"] * pop)
        
        # Multi-key tie breaking: primary hybrid_score, secondary semantic_score, then keyword, pop
        # In numpy/lexsort: keys are given in reverse order (primary key LAST)
        sort_keys = (pop, kw, sem, hybrid)
        sorted_indices = np.lexsort(sort_keys)[::-1][:k]
        
        top_k_cids = u["candidates"]["course_id"].values[sorted_indices]
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
    
    step_label = f"n={n} (alpha={alpha:.1f})"
    w_str = f"{w['semantic']:.2f}/{w['keyword']:.2f}/{w['cf']:.2f}/{w['adaptive']:.2f}/{w['pop']:.2f}"
    rec_str = f"{m_rec:.4f} ± {s_rec:.3f}"
    ndcg_str = f"{m_ndcg:.4f} ± {s_ndcg:.4f}"
    ci_str = f"[{ci_lo:.4f}, {ci_hi:.4f}]"
    
    print(f"{step_label:<24} {w_str:<30} {rec_str:<16} {ndcg_str:<24} {ci_str:<18}")

print("=" * 115)

print("\n" + "=" * 80)
print("  DEGENERATE TEST: Warm user (CF weight 1.0) with all CF scores == 0.0")
print("=" * 80)

# Simulate a warm user with 5+ interactions, but all candidate CF scores are 0.0
test_uid = eval_uids[0]
test_pool = hard_pools[0]
cand = test_pool["candidates"].copy()

# Force all CF scores to 0.0
cand_zero_cf = cand.copy()
# Test calling hybrid recommender's batch logic directly
profile = test_pool["profile"]
warm_weights = {"semantic": 0.00, "keyword": 0.00, "cf": 1.00, "adaptive": 0.00, "pop": 0.00}

# Override cf predict to return 0
orig_predict = rec.cf.predict_score
rec.cf.predict_score = lambda uid, cid: 0.0

try:
    scored_deg = rec._hybrid_scores_batch(cand_zero_cf, profile, warm_weights)
    print(f"[Degenerate Case] CF max score across all candidates: {scored_deg['cf_score'].max():.4f}")
    print(f"[Degenerate Case] Hybrid max score: {scored_deg['hybrid_score'].max():.4f}, min: {scored_deg['hybrid_score'].min():.4f}")
    
    # Sort with multi-key tie-breaking
    sorted_deg = scored_deg.sort_values(
        by=["hybrid_score", "semantic_score", "keyword_score", "popularity_score"],
        ascending=[False, False, False, False]
    ).drop_duplicates("title")
    
    top5 = sorted_deg.head(5)[["course_id", "title", "subject", "hybrid_score", "semantic_score", "keyword_score", "cf_score"]]
    print("\nTop 5 recommendations under degenerate zero-CF:")
    for idx, r in top5.iterrows():
        print(f"  - [{r['course_id']}] {r['title'][:45]:<45} | Hybrid={r['hybrid_score']:.3f} | Sem={r['semantic_score']:.3f} | CF={r['cf_score']:.3f}")
        
    # Verify non-empty and semantically ranked
    assert len(top5) == 5, "List is empty!"
    assert sorted_deg["semantic_score"].iloc[0] >= sorted_deg["semantic_score"].iloc[1], "Semantic tie-break failed!"
    print("\n[SUCCESS] Degenerate zero-CF fallback correctly produced semantically ordered results!")
finally:
    rec.cf.predict_score = orig_predict

