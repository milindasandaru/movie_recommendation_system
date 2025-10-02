Movie Recommendation System
===========================

Lightweight educational movie recommendation system demonstrating three paradigms:

1. Content-Based Filtering (TF-IDF over movie title / genres / tags)
2. Collaborative Filtering (latent factors via TruncatedSVD)
3. Hybrid Blending (weighted combination of content + collaborative signals)

The repository excludes raw CSV datasets (MovieLens or similar) to keep the git history small. Place the required CSV files locally under `data/` as described below.

---

---

Happy recommending! 🚀

## 7. Notebooks

Use the provided notebooks for experimentation:
- `eda.ipynb`: Inspect distributions and sparsity.
- `content_based.ipynb`: Tune TF-IDF or explore alternative text normalization.
- `collaborative.ipynb`: Explore latent dimension impact.

Launch Jupyter:
```powershell
python -m pip install notebook  # if needed
jupyter notebook
```

---

## 8. Removing Unused Dependencies

If you encounter install errors for `scikit-surprise` and do not plan to switch back to it:
```powershell
pip uninstall scikit-surprise
```
Then remove the line from `requirements.txt`.

---

## 9. Potential Extensions

- Popularity or Bayesian average fallback for cold-start users.
- Implicit feedback using play/watch counts.
- Evaluation metrics module (Recall@K, MAP, NDCG, coverage, diversity).
- Caching layer for frequent queries.
- REST API (FastAPI / Flask) wrapper for deployment.

---

## 10. Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| No recommendations (empty list) | Missing CSVs | Ensure files in `data/` and correct names |
| Title not found | Exact string mismatch | Try case-insensitive variant or verify parentheses/year |
| Long runtime on fit | Large dataset | Start with small MovieLens subset or reduce TF-IDF features |
| Memory spike | Attempted full similarity matrix | Current implementation avoids this; ensure you didn't modify code |
| scikit-surprise build failure | Python version / compiler | Remove dependency; not needed now |

---

## 11. License & Data Attribution

Movie data (if using MovieLens) must follow the GroupLens license & citation guidelines. Cite appropriately:

```
F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19 (December 2015), 19 pages.
```

---

## 12. Quick Smoke Test (Without Data)

Runs but yields empty lists (expected):
```powershell
python app.py content --title "Toy Story (1995)" --top 3
```
Output: `No recommendations found.`

After adding data, rerun to see populated results.

---

Happy recommending! 🚀

---

## 13. REST API & Frontend

A FastAPI service (`api.py`) exposes JSON endpoints:

| Endpoint | Params | Description |
|----------|--------|-------------|
| GET /health | - | Health check |
| GET /recommend/content | title, n, explain? | Content-based by title |
| GET /recommend/user | user_id, n, explain? | Collaborative by user |
| GET /recommend/hybrid | user_id, title, alpha, n, explain? | Hybrid blend |

Run the API:
```powershell
uvicorn api:app --reload --port 8000
```
Then open interactive docs at: http://127.0.0.1:8000/docs

### Frontend

Static demo page under `web/index.html` can be served by any static server (or just open directly if same origin with API). For local simplicity:
```powershell
python -m http.server 5500 -d web
```
Open http://127.0.0.1:5500 and ensure the API is running (CORS is enabled).

---

## 14. LLM Explanations (Optional, Open‑Source Only)

`llm_explainer.py` now uses ONLY open-source Hugging Face models (default: `sshleifer/distilbart-cnn-12-6`) for short natural-language rationales. No credit-based APIs are called.

If transformers not installed or the model download fails, a heuristic explanation string is produced instead.

Install (already in `requirements.txt`):
```powershell
pip install transformers sentencepiece accelerate
```

Optional: choose a different summarization model:
```powershell
$Env:LLM_MODEL_ID = "facebook/bart-large-cnn"
```

Use the API with explanation:
```powershell
curl "http://127.0.0.1:8000/recommend/hybrid?user_id=1&title=Toy%20Story%20(1995)&explain=true"
```

No API keys required. Large models may take time and memory; pick a smaller model if constraints apply.

---

## 15. Next Ideas

- Embedding-based semantic expansion (e.g., sentence-transformers) for better cold-start.
- Batch LLM explanation generation + caching.
- Richer frontend (React/Vue) with charts and rating feedback loop.
- Dockerfile + CI/CD pipeline.


