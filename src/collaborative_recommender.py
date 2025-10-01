"""Collaborative filtering recommender (matrix factorization via Surprise).

This module provides a simple wrapper around the Surprise SVD algorithm.
It loads ratings lazily from `data/ratings.csv` (ignored by git), fits an
SVD model, and can produce top-N recommendations for a given user based
on predicted ratings for unseen items.

It also offers a helper to compute a naive precision@K using a content
recommender to generate candidate items (hybrid-ish evaluation example).

Key differences from the previous version:
 - Converted ad-hoc script into a class `CollaborativeRecommender`.
 - Removed global code execution on import (no immediate CSV load).
 - Fixed inconsistent column name typo: used `userId` (MovieLens spec) instead of `userID`.
 - Added defensive checks and comments for clarity.
 - Isolated evaluation logic from training logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split, accuracy

try:  # Optional import if content recommender available
    from .content_recommender import ContentRecommender
except Exception:  # pragma: no cover - soft fail for modular use
    ContentRecommender = None  # type: ignore


class CollaborativeRecommender:
    """Wrapper around Surprise SVD for movie recommendations."""

    def __init__(
        self,
        ratings_path: Path | str = Path("data") / "ratings.csv",
        rating_scale: tuple[float, float] = (0.5, 5.0),
    ) -> None:
        self.ratings_path = Path(ratings_path)
        self.rating_scale = rating_scale
        self._ratings: Optional[pd.DataFrame] = None
        self._algo: Optional[SVD] = None
        self._fitted: bool = False

    # -------------------- Internal utilities -------------------- #
    def _load_ratings(self) -> bool:
        """Load ratings DataFrame if present.

        Returns True if loaded, False if file missing.
        """
        if not self.ratings_path.exists():
            return False
        df = pd.read_csv(self.ratings_path)
        # Basic schema normalization
        expected_cols = {"userId", "movieId", "rating"}
        if not expected_cols.issubset(df.columns):
            raise ValueError(
                f"ratings.csv must contain columns {expected_cols}, found {set(df.columns)}"
            )
        self._ratings = df
        return True

    # -------------------- Public API -------------------- #
    def fit(self) -> None:
        """Train the SVD model using Surprise.

        Loads ratings lazily, splits into train/test for a quick evaluation
        (prints RMSE), then retains the fitted algorithm for inference.
        """
        if not self._load_ratings():  # Ratings missing
            self._fitted = False
            return

        assert self._ratings is not None
        reader = Reader(rating_scale=self.rating_scale)
        data = Dataset.load_from_df(self._ratings[["userId", "movieId", "rating"]], reader)
        trainset, testset = train_test_split(data, test_size=0.2)

        algo = SVD()
        algo.fit(trainset)
        # Quick evaluation (side-effect print) — optional; could be logged
        predictions = algo.test(testset)
        try:
            accuracy.rmse(predictions)
        except Exception:
            pass

        self._algo = algo
        self._fitted = True

    def is_fitted(self) -> bool:
        return self._fitted

    def recommend(self, user_id: int, n: int = 10) -> List[dict]:
        """Recommend top-N movies for a given user based on predicted rating.

        Strategy:
          1. Ensure model is fitted (lazy fit attempted).
          2. Identify movies the user has NOT rated.
          3. Predict ratings for those movies using SVD.
          4. Return the top-N by predicted rating.
        """
        if not self._fitted:
            self.fit()
            if not self._fitted:
                return []

        assert self._ratings is not None and self._algo is not None

        user_rated = set(self._ratings[self._ratings.userId == user_id].movieId)
        all_movie_ids = set(self._ratings.movieId.unique())
        unseen = list(all_movie_ids - user_rated)
        if not unseen:
            return []

        # Predict ratings for unseen movies
        preds = []
        for mid in unseen:
            est = self._algo.predict(uid=user_id, iid=mid).est
            preds.append((mid, est))

        # Sort by estimated rating descending and take top-n
        preds.sort(key=lambda x: x[1], reverse=True)
        top = preds[:n]
        return [
            {"movieId": mid, "predicted_rating": round(score, 3)}
            for mid, score in top
        ]

    # -------------------- Evaluation Helper -------------------- #
    def precision_at_k(
        self,
    user_id: int,
    content_recommender: Optional[object] = None,
        k: int = 10,
        relevance_threshold: float = 4.0,
    ) -> float:
        """Compute a naive precision@k for a user's liked movies.

        If a content-based recommender is supplied, we: pick one liked movie,
        generate K similar titles, and measure how many are also liked.

        NOTE: This is illustrative only and not a rigorous evaluation metric.
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

        # Use first liked movie as anchor
        anchor_id = liked.iloc[0]
        # Find title from content recommender's movie frame if available
        title = None
        if getattr(content_recommender, "_movies", None) is not None:
            df_movies = content_recommender._movies  # type: ignore[attr-defined]
            match = df_movies[df_movies.movieId == anchor_id]
            if not match.empty:
                title = match.iloc[0].title
        if title is None:
            return 0.0

        recs = content_recommender.recommend(title, k)
        rec_ids = {r.get("movieId") for r in recs}
        if not rec_ids:
            return 0.0

        precision = len(set(liked).intersection(rec_ids)) / float(k)
        return precision


if __name__ == "__main__":  # Manual quick smoke test (will be no-op if data missing)
    cr = CollaborativeRecommender()
    print("Fitting collaborative recommender (if data present)...")
    cr.fit()
    if cr.is_fitted():
        sample_user = int(cr._ratings.userId.sample(1).iloc[0])  # type: ignore
        print("Sample recommendations:", cr.recommend(sample_user, 5))
    else:
        print("ratings.csv not found; skipped training.")
