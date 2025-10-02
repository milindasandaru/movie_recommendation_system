"""Collaborative filtering recommender (pure scikit-learn SVD fallback).

This implementation avoids the `scikit-surprise` dependency (which failed to
build under the current Python environment) by using a simple latent factor
model learned via TruncatedSVD over the user-item rating matrix.

Pipeline:
 1. Load ratings (or accept an in-memory DataFrame) with columns userId, movieId, rating.
 2. Build a dense user-item matrix (NaNs -> 0).
 3. Apply TruncatedSVD to obtain user and item latent embeddings.
 4. Reconstruct approximate preference scores (dot product) for ranking.

Notes / Simplifications:
 - Zero-fill for missing ratings biases toward popular items; for a real system
   consider mean-centering or using implicit feedback weighting.
 - No regularization / bias terms; this is strictly educational scaffolding.
 - Designed to fail gracefully (returns empty lists if data missing).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd
from sklearn.decomposition import TruncatedSVD
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

try:  # Optional content recommender reference
    from .content_recommender import ContentRecommender  # type: ignore
except Exception:  # pragma: no cover
    ContentRecommender = None  # type: ignore


class CollaborativeRecommender:
    """Latent-factor collaborative filtering with TruncatedSVD."""

    def __init__(
        self,
        ratings_path: Path | str = Path("data") / "ratings.csv",
        n_components: int = 50,
        min_ratings_user: int = 1,
        min_ratings_item: int = 1,
        max_users: Optional[int] = 20000,
        max_items: Optional[int] = 15000,
        top_items_strategy: str = "popularity",  # or 'ratings_count'
    ) -> None:
        self.ratings_path = Path(ratings_path)
        self.n_components = n_components
        self.min_ratings_user = min_ratings_user
        self.min_ratings_item = min_ratings_item
        self.max_users = max_users
        self.max_items = max_items
        self.top_items_strategy = top_items_strategy

        # Internal artifacts
        self._ratings: Optional[pd.DataFrame] = None
        self._user_index: dict[int, int] = {}
        self._movie_index: dict[int, int] = {}
        self._index_user: list[int] = []
        self._index_movie: list[int] = []
        self._user_factors: Optional[np.ndarray] = None
        self._item_factors: Optional[np.ndarray] = None
    # Large dense score matrix avoided for memory reasons; we compute per-user scores lazily.
        self._fitted = False

    # -------------------- Data Loading -------------------- #
    def _load_ratings(self) -> bool:
        if self._ratings is not None:
            return True
        if not self.ratings_path.exists():
            return False
        df = pd.read_csv(self.ratings_path)
        required = {"userId", "movieId", "rating"}
        if not required.issubset(df.columns):
            raise ValueError(f"ratings data must have columns {required}")
        self._ratings = df
        return True

    # -------------------- Fitting -------------------- #
    def fit(self, ratings_df: Optional[pd.DataFrame] = None) -> None:
        """Fit latent factors.

        Args:
            ratings_df: Optional external DataFrame (userId, movieId, rating).
                        If omitted, tries to load from `ratings_path`.
        """
        if ratings_df is not None:
            self._ratings = ratings_df.copy()
        elif not self._load_ratings():
            self._fitted = False
            return

        assert self._ratings is not None
        df = self._ratings

        # Basic filtering (drop extremely sparse users/items if thresholds set)
        if self.min_ratings_user > 1:
            user_counts = df.groupby('userId').size()
            keep_users = user_counts[user_counts >= self.min_ratings_user].index
            df = df[df.userId.isin(keep_users)]
        if self.min_ratings_item > 1:
            item_counts = df.groupby('movieId').size()
            keep_items = item_counts[item_counts >= self.min_ratings_item].index
            df = df[df.movieId.isin(keep_items)]

        if df.empty:
            self._fitted = False
            return

        # OPTIONAL SAMPLING / LIMITING for scalability ---------------------------------
        # Limit users by activity if exceeding max_users
        if self.max_users is not None:
            user_activity = df.groupby('userId').size().sort_values(ascending=False)
            if len(user_activity) > self.max_users:
                keep_users = set(user_activity.head(self.max_users).index)
                df = df[df.userId.isin(keep_users)]

        # Limit items by popularity / ratings count if exceeding max_items
        if self.max_items is not None:
            item_pop = df.groupby('movieId').agg({'rating': ['count', 'mean']})
            item_pop.columns = ['count', 'mean']
            if len(item_pop) > self.max_items:
                if self.top_items_strategy == 'popularity':
                    top_items = set(item_pop.sort_values('count', ascending=False).head(self.max_items).index)
                else:  # ratings_count alias
                    top_items = set(item_pop.sort_values('count', ascending=False).head(self.max_items).index)
                df = df[df.movieId.isin(top_items)]

        if df.empty:
            self._fitted = False
            return

        # Build sparse matrix (users x movies) ------------------------------------------------
        user_codes = df['userId'].astype('category')
        movie_codes = df['movieId'].astype('category')
        rows = user_codes.cat.codes.to_numpy()
        cols = movie_codes.cat.codes.to_numpy()
        data = df['rating'].astype(float).to_numpy()
        n_users = rows.max() + 1
        n_items = cols.max() + 1

        sparse_matrix: csr_matrix = coo_matrix((data, (rows, cols)), shape=(n_users, n_items)).tocsr()

        # Store mapping back to real IDs
        self._index_user = [int(u) for u in user_codes.cat.categories]
        self._index_movie = [int(m) for m in movie_codes.cat.categories]
        self._user_index = {u: i for i, u in enumerate(self._index_user)}
        self._movie_index = {m: i for i, m in enumerate(self._index_movie)}

        # Adjust component count (cannot exceed min(n_users, n_items) - 1)
        max_components = max(1, min(n_users - 1, n_items - 1))
        n_comp = min(self.n_components, max_components)

        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        user_factors = svd.fit_transform(sparse_matrix)  # (num_users, k)
        item_factors = svd.components_.T  # (num_items, k)

        self._user_factors = user_factors
        self._item_factors = item_factors
        self._fitted = True

    def is_fitted(self) -> bool:
        return self._fitted

    # -------------------- Recommendation -------------------- #
    def recommend(self, user_id: int, n: int = 10) -> List[dict]:
        """Return top-N movie recommendations for a user.

        Strategy:
          * Lazy fit if needed.
          * If user unseen -> return empty list (could fallback to popular items).
          * Rank by predicted latent score excluding already-rated items.
        """
        if not self._fitted:
            self.fit()
            if not self._fitted:
                return []

        assert self._ratings is not None
        if user_id not in self._user_index:
            return []

        uidx = self._user_index[user_id]
        assert self._user_factors is not None and self._item_factors is not None
        user_vector = self._user_factors[uidx]
        # Compute scores lazily: dot with all item factors
        user_scores = user_vector @ self._item_factors.T  # shape (num_items,)

        # Exclude rated items
        rated_items = set(self._ratings[self._ratings.userId == user_id].movieId)

        candidates = []
        for mid, midx in self._movie_index.items():
            if mid in rated_items:
                continue
            candidates.append((mid, user_scores[midx]))

        if not candidates:
            return []

        candidates.sort(key=lambda x: x[1], reverse=True)
        top = candidates[:n]
        return [
            {"movieId": mid, "score": round(float(score), 4)}
            for mid, score in top
        ]

    # -------------------- Precision@K (Illustrative) -------------------- #
    def precision_at_k(
        self,
        user_id: int,
        content_recommender: Optional[object] = None,
        k: int = 10,
        relevance_threshold: float = 4.0,
    ) -> float:
        """Very naive precision@k using content-based similar titles.

        Not a rigorous evaluation; for demonstration only.
        """
        if self._ratings is None:
            if not self._load_ratings():
                return 0.0
        assert self._ratings is not None
        user_rows = self._ratings[self._ratings.userId == user_id]
        liked = user_rows[user_rows.rating >= relevance_threshold]["movieId"]
        if liked.empty:
            return 0.0
        if content_recommender is None or not hasattr(content_recommender, "recommend"):
            return 0.0
        anchor = liked.iloc[0]
        # Attempt to map anchor to title via content recommender
        title = None
        movies_df = getattr(content_recommender, "_movies", None)
        if movies_df is not None:
            match = movies_df[movies_df.movieId == anchor]
            if not match.empty:
                title = match.iloc[0].title
        if title is None:
            return 0.0
        recs = content_recommender.recommend(title, k)
        rec_ids = {r.get("movieId") for r in recs}
        if not rec_ids:
            return 0.0
        return len(set(liked).intersection(rec_ids)) / float(k)


if __name__ == "__main__":  # Simple self-test with synthetic data
    # Build a tiny synthetic rating set if real data missing
    synth = pd.DataFrame({
        'userId': [1, 1, 1, 2, 2, 3, 3, 4],
        'movieId': [10, 11, 12, 10, 13, 11, 14, 12],
        'rating': [5, 4, 3, 4, 5, 2, 4, 5]
    })
    cr = CollaborativeRecommender()
    cr.fit(synth)
    print("Recommendations for user 1:", cr.recommend(1, 5))
