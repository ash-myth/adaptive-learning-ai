"""
Test ranking evaluation with session/subject topic context for held-out interaction.
"""
import time
import numpy as np
import pandas as pd
from hybrid_recommender import HybridRecommender

rec = HybridRecommender(retrain=False)
users_df = pd.read_csv("data/users.csv")
interactions_df = pd.read_csv("data/interactions.csv")

completed_users = interactions_df[interactions_df["completed"] == 1].groupby("user_id").size()
eligible_users = completed_users[completed_users >= 3].index.tolist()

rng = np.random.default_rng(42)
sample_size = min(100, len(eligible_users))
eval_uids = rng.choice(eligible_users, size=sample_size, replace=False)
print(f"Sampled {len(eval_uids)} users.")

user_eval_data = []

t0 = time.time()
for uid in eval_uids:
    u_iacts = interactions_df[interactions_df["user_id"] == uid]
    u_comp = u_iacts[u_iacts["completed"] == 1]
    
    # Held-out course
    held_out_cid = str(u_comp.iloc[-1]["course_id"])
    train_seen = set(u_iacts["course_id"].astype(str)) - {held_out_cid}
    
    held_row = rec.courses_df[rec.courses_df["course_id"] == held_out_cid].iloc[0]
    held_subj = str(held_row["subject"])
    
    # Candidate pool for the subject topic
    candidates = rec._candidate_pool(held_subj, size=150).copy()
    
    # Ensure held_out_cid is in candidate pool
    if held_out_cid not in candidates["course_id"].values:
        h_df = pd.DataFrame([held_row])
        # Compute exact semantic and BM25 scores for held_out
        expanded = rec.embed_engine._expand(held_subj)
        qv = rec.embed_engine.encoder.transform([expanded])[0].reshape(1, -1)
        h_idx = rec.courses_df[rec.courses_df["course_id"] == held_out_cid].index[0]
        sim = float(np.dot(qv, rec.embed_engine.embeddings[h_idx]) / (
            np.linalg.norm(qv) * np.linalg.norm(rec.embed_engine.embeddings[h_idx]) + 1e-9))
        bm_score = float(rec.embed_engine.bm25.score(held_subj)[h_idx])
        h_df["_sem_sim"] = max(0.0, sim)
        h_df["_bm25_raw"] = max(0.0, bm_score)
        candidates = pd.concat([candidates, h_df], ignore_index=True)

    # Filter out other seen training courses
    candidates["course_id"] = candidates["course_id"].astype(str)
    candidates = candidates[~candidates["course_id"].isin(train_seen)].copy()
    
    if held_out_cid not in candidates["course_id"].values:
        continue

    profile = rec._get_profile(uid)
    bm25_max = float(candidates["_bm25_raw"].max()) or 1.0
    profile["_bm25_max"] = bm25_max
    
    sem = np.clip(candidates["_sem_sim"].values.astype(float), 0, None)
    kw = candidates["_bm25_raw"].values.astype(float) / (bm25_max + 1e-9)
    
    cf_raw = np.array([rec.cf.predict_score(uid, cid) for cid in candidates["course_id"]], dtype=float)
    cf = np.clip(cf_raw, 0.0, 1.0)
    
    ranked = rec.adaptive.rank_courses(profile, candidates, top_n=len(candidates))
    adap_map = dict(zip(ranked["course_id"].astype(str), ranked["adaptive_score"]))
    adap = np.array([adap_map.get(cid, 0.0) for cid in candidates["course_id"]], dtype=float)
    
    pop = candidates["popularity_score"].fillna(0.5).values.astype(float)
    cids = candidates["course_id"].values
    
    user_eval_data.append({
        "user_id": uid,
        "target": held_out_cid,
        "cids": cids,
        "sem": sem,
        "kw": kw,
        "cf": cf,
        "adap": adap,
        "pop": pop,
    })

print(f"Precomputed scores for {len(user_eval_data)} users in {time.time()-t0:.1f}s.")

configs = [
    ("Popularity Only", {"semantic": 0.00, "keyword": 0.00, "cf": 0.00, "adaptive": 0.00, "pop": 1.00}),
    ("CF Only",         {"semantic": 0.00, "keyword": 0.00, "cf": 1.00, "adaptive": 0.00, "pop": 0.00}),
    ("Semantic Only",   {"semantic": 1.00, "keyword": 0.00, "cf": 0.00, "adaptive": 0.00, "pop": 0.00}),
    ("Current Default", {"semantic": 0.30, "keyword": 0.25, "cf": 0.25, "adaptive": 0.15, "pop": 0.05}),
    ("Config A (More CF 0.35)",  {"semantic": 0.20, "keyword": 0.20, "cf": 0.35, "adaptive": 0.15, "pop": 0.10}),
    ("Config B (More CF 0.45)",  {"semantic": 0.20, "keyword": 0.15, "cf": 0.45, "adaptive": 0.15, "pop": 0.05}),
    ("Config C (More Sem 0.40)", {"semantic": 0.40, "keyword": 0.20, "cf": 0.20, "adaptive": 0.15, "pop": 0.05}),
    ("Config D (Balanced 0.25)", {"semantic": 0.25, "keyword": 0.25, "cf": 0.25, "adaptive": 0.20, "pop": 0.05}),
    ("Config E (CF 0.30 + Adap 0.20)", {"semantic": 0.25, "keyword": 0.20, "cf": 0.30, "adaptive": 0.20, "pop": 0.05}),
]

k = 10
print("\n" + "=" * 76)
print(f"{'Configuration':<30} {'Recall@10':<12} {'Prec@10':<12} {'NDCG@10':<12}")
print("=" * 76)

for name, w in configs:
    recalls, precs, ndcgs = [], [], []
    for u in user_eval_data:
        scores = (
            w["semantic"] * u["sem"]
            + w["keyword"] * u["kw"]
            + w["cf"] * u["cf"]
            + w["adaptive"] * u["adap"]
            + w["pop"] * u["pop"]
        )
        top_indices = np.argsort(scores)[::-1][:k]
        top_cids = u["cids"][top_indices]
        target = u["target"]
        if target in top_cids:
            rank = int(np.where(top_cids == target)[0][0]) + 1
            recalls.append(1.0)
            precs.append(1.0 / k)
            ndcgs.append(1.0 / np.log2(rank + 1))
        else:
            recalls.append(0.0)
            precs.append(0.0)
            ndcgs.append(0.0)
            
    print(f"{name:<30} {np.mean(recalls):<12.4f} {np.mean(precs):<12.4f} {np.mean(ndcgs):<12.4f}")
print("=" * 76)

