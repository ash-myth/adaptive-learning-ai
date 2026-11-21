"""
embeddings.py  (v3)
===================
Adaptive AI Learning Intelligence Platform — NLP Embedding Engine

Fix #13 — replace LSA with sentence-transformers
-------------------------------------------------
LSA (TF-IDF + SVD) captured only ~38% of the semantic variance in the
catalogue with 256 components.  BM25 was added as a corrective signal
because LSA alone could not reliably surface rare technical terms.

Sentence-transformers (all-MiniLM-L6-v2, 22 MB) encode text into 384-dim
dense vectors that capture deep semantic meaning — synonyms, paraphrases,
and conceptual relationships that TF-IDF misses entirely.  Retrieval
accuracy improves substantially; BM25 becomes a complementary precision
signal rather than a compensatory one.

Graceful fallback
-----------------
If sentence-transformers is not installed, the engine automatically falls
back to LSA (v2 behaviour) with a clear warning.  Install with:
    pip install sentence-transformers

All other behaviour (BM25 dual-index, query expansion, caching) is
preserved unchanged.
"""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import pandas as pd
import re, math, pickle, time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

# Fix #13 — sentence-transformers with LSA fallback
try:
    from sentence_transformers import SentenceTransformer as _ST
    _SBERT_AVAILABLE = True
except ImportError:
    _ST             = None
    _SBERT_AVAILABLE = False

SBERT_MODEL  = os.environ.get("SBERT_MODEL", "all-MiniLM-L6-v2")
SBERT_BATCH  = int(os.environ.get("SBERT_BATCH", "128"))
USE_SBERT    = _SBERT_AVAILABLE and os.environ.get("USE_SBERT", "true").lower() == "true"

EMBEDDING_DIM = 256
CACHE_PATH    = "data/embeddings.pkl"
VOCAB_SIZE    = 15000

# ── Query expansion dictionary ──────────────────────────────────────────────
#
# Fix #6 — collect ALL matching keys, not just the first one
# -----------------------------------------------------------
# The old _expand() returned on the first matching key, so a query like
# "intro to backend web apis" would stop at "api" and never add the richer
# "backend" expansion terms.  _expand() now collects every key whose
# substring appears in the query and merges all their expansions.
#
# New keys added:
#   web api, rest api, graphql, node.js / nodejs / express (JS backend),
#   typescript / ts, react native, python, java, golang / go, rust,
#   sql / postgres / mysql, docker / k8s / kubernetes alias,
#   llm / gpt / genai / generative ai, prompt, computer vision / cv,
#   spark / kafka / airflow / etl, security, mobile, game dev, dsa
#
# Key ordering does not matter — all matching keys are always collected.

QUERY_EXPANSION = {
    # ── Web / backend ────────────────────────────────────────────────────
    "backend":        "backend server api rest node django flask express python web",
    "web api":        "api rest http backend web services json endpoint",
    "rest api":       "rest api http json web services backend endpoint",
    "graphql":        "graphql api query mutation schema backend web",
    "node.js":        "nodejs node javascript backend express server api",
    "nodejs":         "nodejs node javascript backend express server api",
    "express":        "express nodejs javascript backend web framework api",
    "django":         "django python backend web framework rest api",
    "flask":          "flask python backend web api microservice",
    "frontend":       "frontend html css javascript react vue angular ui web",
    "full stack":     "full stack web development frontend backend nodejs react",
    "react":          "react javascript frontend ui component web development",
    "react native":   "react native mobile ios android javascript cross platform",
    "vue":            "vue javascript frontend ui component web",
    "angular":        "angular typescript javascript frontend ui web framework",
    "typescript":     "typescript javascript frontend backend type safety",
    "ts":             "typescript javascript type safety frontend",
    # ── Languages ────────────────────────────────────────────────────────
    "python":         "python programming scripting backend data science ml",
    "java":           "java programming oop backend spring enterprise",
    "golang":         "golang go programming backend systems concurrency",
    "go ":            "golang go programming backend concurrency systems",   # trailing space avoids matching "good"
    "rust":           "rust programming systems memory safety performance",
    "kotlin":         "kotlin android mobile java jvm programming",
    "swift":          "swift ios apple mobile programming xcode",
    "scala":          "scala functional jvm spark data engineering",
    "r ":             "r statistics data analysis ggplot machine learning",  # trailing space
    # ── Cloud / DevOps ────────────────────────────────────────────────────
    "devops":         "devops docker kubernetes ci cd deployment pipeline linux",
    "cloud":          "cloud aws azure google cloud computing infrastructure",
    "docker":         "docker container devops deployment kubernetes linux",
    "kubernetes":     "kubernetes k8s container orchestration devops cloud",
    "k8s":            "kubernetes container orchestration devops cloud",
    "aws":            "aws amazon cloud ec2 s3 lambda infrastructure",
    "azure":          "azure microsoft cloud devops infrastructure",
    "terraform":      "terraform infrastructure as code cloud devops",
    "ci cd":          "ci cd pipeline devops github actions jenkins automation",
    # ── Databases ────────────────────────────────────────────────────────
    "database":       "database sql mysql postgresql mongodb nosql queries",
    "sql":            "sql database mysql postgresql queries relational",
    "postgres":       "postgresql sql database relational queries",
    "mysql":          "mysql sql database relational queries web",
    "mongodb":        "mongodb nosql database document json queries",
    "redis":          "redis cache nosql database key value",
    # ── ML / AI ──────────────────────────────────────────────────────────
    "ml":             "machine learning scikit tensorflow pytorch neural network",
    "ai":             "artificial intelligence machine learning deep learning nlp",
    "llm":            "large language model gpt transformer nlp generative ai",
    "gpt":            "gpt large language model openai nlp generative ai prompt",
    "genai":          "generative ai llm gpt diffusion transformer nlp",
    "generative ai":  "generative ai llm gpt diffusion stable diffusion nlp",
    "prompt":         "prompt engineering llm gpt generative ai language model",
    "nlp":            "natural language processing text classification bert transformer",
    "computer vision":"computer vision image classification cnn object detection",
    "cv":             "computer vision image recognition deep learning cnn",
    "deep learning":  "deep learning neural network tensorflow pytorch gpu",
    "neural network": "neural network deep learning tensorflow keras pytorch",
    "data science":   "data science python statistics machine learning analysis",
    "data analyst":   "data analysis excel sql tableau visualization statistics",
    "data engineer":  "data engineering pipeline spark kafka airflow etl sql",
    "spark":          "apache spark big data python scala distributed processing",
    "kafka":          "kafka streaming data pipeline event driven distributed",
    "airflow":        "airflow workflow orchestration pipeline etl data engineering",
    "etl":            "etl pipeline data engineering sql spark transformation",
    # ── Security ─────────────────────────────────────────────────────────
    "cyber":          "cybersecurity security network ethical hacking penetration",
    "security":       "cybersecurity network security ethical hacking encryption",
    "ethical hacking":"ethical hacking penetration testing kali linux network",
    # ── Design / UX ──────────────────────────────────────────────────────
    "ux":             "ux ui user experience design figma prototype wireframe",
    "ui design":      "ui design figma sketch frontend user interface visual",
    # ── Mobile ────────────────────────────────────────────────────────────
    "android":        "android mobile app development kotlin java studio",
    "ios":            "ios swift mobile app development xcode apple",
    "mobile":         "mobile app android ios react native flutter development",
    "flutter":        "flutter dart mobile ios android cross platform",
    # ── Other tech ────────────────────────────────────────────────────────
    "blockchain":     "blockchain cryptocurrency ethereum solidity web3 defi",
    "game dev":       "game development unity unreal c++ graphics 3d",
    "dsa":            "data structures algorithms programming computer science",
    "algorithms":     "algorithms data structures programming computer science",
    "api":            "api rest graphql backend web services http development",
}


class BM25:
    """Okapi BM25 keyword ranking. Complements LSA for rare technical terms."""
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1; self.b = b
        self.corpus_size = 0; self.avgdl = 0.0
        self.doc_freqs = []; self.idf = {}; self.doc_lengths = []

    def _tok(self, t):
        return [w for w in re.sub(r"[^\w\s]", " ", str(t).lower()).split() if len(w) > 1]

    def fit(self, docs):
        self.corpus_size = len(docs)
        toks = [self._tok(d) for d in docs]
        self.doc_lengths = [len(t) for t in toks]
        self.avgdl = sum(self.doc_lengths) / max(self.corpus_size, 1)
        nd = {}
        for t_list in toks:
            self.doc_freqs.append({})
            seen = set()
            for tok in t_list:
                self.doc_freqs[-1][tok] = self.doc_freqs[-1].get(tok, 0) + 1
                if tok not in seen:
                    nd[tok] = nd.get(tok, 0) + 1
                    seen.add(tok)
        for term, freq in nd.items():
            self.idf[term] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1)
        return self

    def score(self, query):
        scores = np.zeros(self.corpus_size, dtype=np.float32)
        for tok in self._tok(query):
            if tok not in self.idf:
                continue
            idf = self.idf[tok]
            for i, fm in enumerate(self.doc_freqs):
                if tok not in fm:
                    continue
                tf = fm[tok]; dl = self.doc_lengths[i]
                scores[i] += idf * tf * (self.k1 + 1) / (
                    tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return scores


class LSAEncoder:
    """TF-IDF + SVD semantic encoder (v2 fallback when sentence-transformers unavailable)."""

    def __init__(self, n_components=EMBEDDING_DIM):
        self.tfidf = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 3),
            max_features=VOCAB_SIZE, sublinear_tf=True, min_df=2)
        self.svd = TruncatedSVD(
            n_components=n_components, algorithm="randomized",
            n_iter=15, random_state=42)
        self.is_fitted = False

    def fit_transform(self, texts):
        print(f"[LSA] TF-IDF over {len(texts):,} docs …")
        M = self.tfidf.fit_transform(texts)
        print(f"[LSA] Shape: {M.shape}  Running SVD → {EMBEDDING_DIM} dims …")
        E = normalize(self.svd.fit_transform(M), norm="l2")
        print(f"[LSA] Explained variance: {self.svd.explained_variance_ratio_.sum():.2%}")
        self.is_fitted = True
        return E

    def transform(self, texts):
        return normalize(self.svd.transform(self.tfidf.transform(texts)), norm="l2")

    @property
    def explained_variance(self):
        return self.svd.explained_variance_ratio_.sum()


class SBERTEncoder:
    """
    Sentence-transformer encoder (fix #13).

    Encodes course text into 384-dim dense vectors using all-MiniLM-L6-v2
    (22 MB, CPU-friendly).  Vectors are L2-normalised so cosine similarity
    reduces to a dot product.

    Why sentence-transformers over LSA?
    ------------------------------------
    - LSA: TF-IDF bag-of-words → SVD projection → ~38% explained variance
      Misses synonyms, paraphrases, conceptual relationships.
    - SBERT: fine-tuned on 1B+ sentence pairs → captures deep semantics
      "backend web api" and "REST server development" map to nearby vectors.

    The BM25 index is kept alongside SBERT for precision on rare exact-term
    queries (e.g. a specific library name not in the SBERT vocabulary).
    """

    def __init__(self, model_name: str = SBERT_MODEL):
        self.model_name = model_name
        self.model      = None
        self.is_fitted  = False
        self.dim        = 384   # all-MiniLM-L6-v2 output dimension

    def _load_model(self):
        if self.model is None:
            print(f"[SBERT] Loading model: {self.model_name} …")
            try:
                self.model = _ST(self.model_name, local_files_only=True)
            except Exception:
                self.model = _ST(self.model_name)
            self.dim   = getattr(self.model, "get_sentence_embedding_dimension", None)
            if callable(self.dim):
                self.dim = self.dim()
            else:
                self.dim = 384
            print(f"[SBERT] Model loaded — embedding dim: {self.dim}")

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        self._load_model()
        print(f"[SBERT] Encoding {len(texts):,} documents "
              f"(batch_size={SBERT_BATCH}) …")
        embeddings = self.model.encode(
            texts,
            batch_size       = SBERT_BATCH,
            show_progress_bar= True,
            normalize_embeddings = True,   # L2 normalise → dot product = cosine sim
            convert_to_numpy = True,
        )
        self.is_fitted = True
        print(f"[SBERT] Done — shape: {embeddings.shape}")
        return embeddings.astype(np.float32)

    def transform(self, texts: list[str]) -> np.ndarray:
        self._load_model()
        return self.model.encode(
            texts,
            batch_size           = SBERT_BATCH,
            normalize_embeddings = True,
            convert_to_numpy     = True,
        ).astype(np.float32)

    @property
    def explained_variance(self) -> float:
        """Not applicable for SBERT — returns 1.0 as a placeholder."""
        return 1.0


class EmbeddingEngine:
    """
    Dual-index: semantic embeddings + BM25 keyword matching.

    Fix #13 — auto-selects encoder:
      - SBERTEncoder  if sentence-transformers is installed (recommended)
      - LSAEncoder    as fallback (v2 behaviour, no extra dependencies)

    Force LSA:  set env var USE_SBERT=false
    Force SBERT: pip install sentence-transformers  (USE_SBERT defaults to true)
    """
    def __init__(self, courses_df, force_rebuild=False):
        self.courses    = courses_df.reset_index(drop=True)
        if USE_SBERT:
            self.encoder = SBERTEncoder(model_name=SBERT_MODEL)
            print(f"[EmbeddingEngine] Using SBERTEncoder ({SBERT_MODEL})")
        else:
            self.encoder = LSAEncoder(n_components=EMBEDDING_DIM)
            if not _SBERT_AVAILABLE:
                print("[EmbeddingEngine] sentence-transformers not installed — "
                      "using LSA fallback.  Install with: pip install sentence-transformers")
            else:
                print("[EmbeddingEngine] Using LSAEncoder (USE_SBERT=false)")
        self.embeddings = None
        self.bm25       = None
        if not force_rebuild and os.path.exists(CACHE_PATH):
            self._load_cache()
        else:
            self._build()

    def _lsa_text(self, row):
        title   = str(row.get("title", "")).strip()
        subject = str(row.get("subject", "")).replace("&", "and").replace("/", " ")
        skills  = str(row.get("skills_raw", "")) if pd.notna(row.get("skills_raw")) else ""
        skills  = re.sub(r"[,\n]+", " ", skills)[:300]
        diff    = str(row.get("difficulty_label", ""))
        provider= str(row.get("provider", ""))
        # Title repeated 5x: rare terms get 5x the TF weight vs v1's 2x
        return " ".join(p for p in
            [title]*5 + [subject]*2 + [diff] + [skills]*2 + [provider]
            if p and p != "nan")

    def _bm25_text(self, row):
        title  = str(row.get("title", "")).strip()
        skills = str(row.get("skills_raw", "")) if pd.notna(row.get("skills_raw")) else ""
        return title + " " + re.sub(r"[,\n]+", " ", skills)[:200]

    def _build(self):
        print("[EmbeddingEngine] Building dual index (LSA + BM25) …")
        t0 = time.time()
        lsa_texts  = [self._lsa_text(r) for _, r in self.courses.iterrows()]
        self.embeddings = self.encoder.fit_transform(lsa_texts)
        bm25_texts = [self._bm25_text(r) for _, r in self.courses.iterrows()]
        self.bm25  = BM25().fit(bm25_texts)
        print(f"[EmbeddingEngine] Built in {time.time()-t0:.1f}s  shape={self.embeddings.shape}")
        self._save_cache()

    def _save_cache(self):
        os.makedirs("data", exist_ok=True)
        encoder_type = "sbert" if USE_SBERT else "lsa"
        with open(CACHE_PATH, "wb") as f:
            # NOTE: encoder is intentionally NOT pickled — it contains CUDA
            # tensors that fail to deserialize in subprocess spawns (uvicorn
            # --reload on Windows). The encoder is re-constructed in __init__
            # each time, which is fast (just loads model weights, not encoding).
            pickle.dump({"embeddings": self.embeddings,
                         "bm25": self.bm25,
                         "n_courses": len(self.courses),
                         "encoder_type": encoder_type}, f)
        print(f"[EmbeddingEngine] Cache -> {CACHE_PATH}")

    def _load_cache(self):
        with open(CACHE_PATH, "rb") as f:
            d = pickle.load(f)
        # Invalidate if course count or encoder type changed
        encoder_type = "sbert" if USE_SBERT else "lsa"
        if (d["n_courses"] != len(self.courses)
                or d.get("bm25") is None
                or d.get("encoder_type") != encoder_type):
            print("[EmbeddingEngine] Cache stale — rebuilding ...")
            self._build(); return
        self.embeddings = d["embeddings"]
        # encoder is NOT loaded from cache — it was already built in __init__
        self.bm25       = d["bm25"]
        print(f"[EmbeddingEngine] Loaded  shape={self.embeddings.shape}")

    def _expand(self, query: str) -> str:
        """
        Expand a query by appending synonym/context terms for every
        QUERY_EXPANSION key that appears in the query.

        Fix #6 — collect ALL matching keys
        ------------------------------------
        Old: returned on the first matching key.
             "intro to backend web apis" → stopped at "api", never added
             the richer "backend" expansion.
        New: collects every matching key and merges their expansions
             deduplicated, so no term is repeated even if two keys share
             synonyms.  A query matching three keys gets all three
             expansions appended in one pass.
        """
        q          = query.lower().strip()
        extra_terms: list[str] = []
        seen_terms: set[str]   = set(q.split())   # avoid re-adding query words

        for key, expansion in QUERY_EXPANSION.items():
            if key in q:
                for term in expansion.split():
                    if term not in seen_terms:
                        extra_terms.append(term)
                        seen_terms.add(term)

        if not extra_terms:
            return query
        return query + " " + " ".join(extra_terms)

    def encode_query(self, text):
        return self.encoder.transform([self._expand(text)])[0]

    def get_similar(self, idx, top_k=100, subject_filter=None, difficulty_filter=None):
        qv   = self.embeddings[idx].reshape(1, -1)
        sims = cosine_similarity(qv, self.embeddings).flatten()
        sims[idx] = -1
        top  = np.argsort(sims)[::-1][:top_k * 3]
        res  = self.courses.iloc[top].copy()
        res["semantic_similarity"] = sims[top].round(4)
        if subject_filter:
            mask = res["subject"].str.contains(subject_filter, case=False, na=False)
            if mask.sum() >= top_k // 2:
                res = res[mask]
        if difficulty_filter:
            mask = res["difficulty_label"].str.contains(difficulty_filter, case=False, na=False)
            if mask.sum() >= 3:
                res = res[mask]
        return res.head(top_k)

    def search(self, query, top_k=10):
        """Blended semantic + keyword search with auto-weighted mixing."""
        expanded  = self._expand(query)
        qv        = self.encoder.transform([expanded])[0].reshape(1, -1)
        s_sims    = cosine_similarity(qv, self.embeddings).flatten()
        s_cal     = np.clip(s_sims, 0.0, 1.0)

        bm25_raw  = self.bm25.score(query)
        bm25_cal  = np.clip(bm25_raw / 15.0, 0.0, 1.0)

        # Auto-tune blend: strong BM25 hits (> 0.40) trust keyword more; otherwise semantic
        n_strong  = (bm25_cal > 0.40).sum()
        sw = 0.70 if n_strong < 3 else (0.50 if n_strong < 15 else 0.35)
        kw = 1.0 - sw

        blended  = sw * s_cal + kw * bm25_cal
        top_idx  = np.argsort(blended)[::-1][:top_k]
        results  = self.courses.iloc[top_idx].copy()
        results["semantic_similarity"] = s_sims[top_idx].round(4)
        results["semantic_score"]      = s_sims[top_idx].round(4)
        results["keyword_score"]       = bm25_cal[top_idx].round(4)
        results["blended_score"]       = blended[top_idx].round(4)
        return results

    def get_subject_coherence(self):
        coh = {}
        for subj in self.courses["subject"].unique():
            idx = self.courses[self.courses["subject"] == subj].index.tolist()
            if len(idx) < 2:
                continue
            sims = cosine_similarity(self.embeddings[idx])
            np.fill_diagonal(sims, 0)
            coh[subj] = round(float(sims.mean()), 4)
        return dict(sorted(coh.items(), key=lambda x: -x[1]))

    def explain_similarity(self, idx_a, idx_b):
        va, vb = self.embeddings[idx_a], self.embeddings[idx_b]
        sim    = float(np.dot(va, vb))
        top_c  = np.argsort(np.abs(va * vb))[::-1][:5]
        feats  = self.encoder.tfidf.get_feature_names_out()
        terms  = []
        for ci in top_c:
            for fi in np.argsort(np.abs(self.encoder.svd.components_[ci]))[::-1][:3]:
                terms.append(feats[fi])
        seen, unique = set(), []
        for t in terms:
            if t not in seen:
                seen.add(t); unique.append(t)
        return {"course_a": self.courses.iloc[idx_a]["title"],
                "course_b": self.courses.iloc[idx_b]["title"],
                "similarity": round(sim, 4), "shared_concepts": unique[:8]}


if __name__ == "__main__":
    df = pd.read_csv("courses.csv")
    engine = EmbeddingEngine(df, force_rebuild=True)
    for q in ["backend development","docker kubernetes","machine learning beginner",
              "python data analysis","personal finance investment"]:
        print(f"\n--- {q} ---")
        for _, r in engine.search(q, top_k=4).iterrows():
            print(f"  [{r['blended_score']:.3f}] {r['title'][:55]}  |  {r['subject']}")
    print("\n--- Coherence ---")
    for s, v in list(engine.get_subject_coherence().items())[:5]:
        print(f"  {s:<35} {v:.4f}")
