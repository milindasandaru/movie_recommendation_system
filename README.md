Movie Recommendation System
===========================

Lightweight educational movie recommendation system demonstrating three paradigms:

1. Content-Based Filtering (TF-IDF over movie title / genres / tags)
2. Collaborative Filtering (latent factors via TruncatedSVD)
3. Hybrid Blending (weighted combination of content + collaborative signals)

The repository excludes raw CSV datasets (MovieLens or similar) to keep the git history small. Place the required CSV files locally under `data/` as described below.

---

## 1. Project Structure

```
movie_recommendation_system/
	app.py                    # CLI entrypoint
	requirements.txt          # Python dependencies
	README.md                 # This file
	data/                     # (Ignored) Put movies.csv, ratings.csv, tags.csv here
		README.md               # Instructions for obtaining datasets
	src/
		content_recommender.py  # Content-based model
		collaborative_recommender.py # Collaborative (TruncatedSVD) model
		hybrid_recommender.py   # Hybrid orchestrator
	notebooks/
		eda.ipynb               # Exploratory Data Analysis starter
		content_based.ipynb     # Content model experimentation
		collaborative.ipynb     # Collaborative model experimentation
```

---

## 2. Data Requirements

Minimum CSV files expected (MovieLens naming convention):

| File         | Required For              | Mandatory | Expected Columns (subset)                 |
|--------------|---------------------------|-----------|-------------------------------------------|
| movies.csv   | Content + Hybrid + Enrich | Yes       | movieId, title, genres                    |
| ratings.csv  | Collaborative + Hybrid    | Yes (for user/hybrid modes) | userId, movieId, rating     |
| tags.csv     | Content (improves text)   | Optional  | userId, movieId, tag                      |

Place them in `data/` (not tracked by git). See `data/README.md` for download instructions (e.g. MovieLens 25M/Latest). Larger datasets (e.g. 25M) may take longer to build TF-IDF; start with the small (e.g. ml-latest-small) subset for experimentation.

---

## 3. Environment Setup

Python version: 3.10–3.12 recommended (avoid 3.13 if `scikit-surprise` install issues arise; current code does NOT require scikit-surprise at runtime).

1. Create and activate a virtual environment:
	 - Windows (PowerShell):
		 ```powershell
		 python -m venv .venv
		 .venv\Scripts\Activate.ps1
		 ```
2. Install dependencies:
	 ```powershell
	 pip install -r requirements.txt
	 ```

Note: `scikit-surprise` is listed but currently unused after refactoring to pure scikit-learn. You may remove it if build issues occur.

---

## 4. CLI Usage (`app.py`)

Run the CLI from the project root after placing the CSVs in `data/`.

General pattern:
```powershell
python app.py <mode> [options]
```

Supported modes:

1. Content-Based by Title:
```powershell
python app.py content --title "Toy Story (1995)" --top 5
```

2. Collaborative by User ID:
```powershell
python app.py user --user 1 --top 10
```

3. Hybrid (User + Anchor Title):
```powershell
python app.py hybrid --user 1 --title "Toy Story (1995)" --alpha 0.6 --top 8
```

Additional global options:
| Option       | Meaning                                |
|--------------|-----------------------------------------|
| --data-dir   | Alternate path to data directory        |
| --json       | Output raw JSON (machine-friendly)      |

Example JSON output:
```powershell
python app.py content --title "Toy Story (1995)" --top 3 --json
```

---

## 5. Programmatic Usage

```python
from src.content_recommender import ContentRecommender
from src.collaborative_recommender import CollaborativeRecommender
from src.hybrid_recommender import HybridRecommender

content = ContentRecommender("data/movies.csv", "data/tags.csv")
content.fit()
print(content.recommend("Toy Story (1995)", 5))

collab = CollaborativeRecommender("data/ratings.csv")
collab.fit()
print(collab.recommend(user_id=1, n=5))

hybrid = HybridRecommender(alpha=0.5)
hybrid.fit()
print(hybrid.recommend_hybrid(user_id=1, title="Toy Story (1995)", n=5))
```

---

## 6. Design Notes & Trade-offs

Content-Based:
- Computes TF-IDF matrix once. Similarity for a single title is computed on-demand (no full NxN cosine matrix to save memory).
- Uses title + genres + aggregated tags.

Collaborative:
- Matrix factorization with `TruncatedSVD` over (user x movie) ratings pivot.
- Zero-filling missing ratings is simplistic; better approaches include centering, implicit feedback weighting, ALS, or using libraries like `implicit`.

Hybrid:
- Inverse-rank scaling of content list combined with raw latent scores.
- Blend weight alpha in [0,1].

---

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

