"""Hybrid recommender combining Content-based and Collaborative filtering.

Supports:
 - User-based recommendations (via collaborative filtering).
 - Title-based recommendations (via content filtering).
 - Blended scoring: alpha * content_score + (1 - alpha) * collab_score.

This is a simple orchestrator that delegates to the respective recommenders.
"""

from __future__ import annotations
from typing import List, Dict, Optional

from pathlib import Path
import pandas as pd

from .content_recommender import ContentRecommender
from .collaborative_recommender import CollaborativeRecommender


class HybridRecommender:
    """Hybrid recommender system."""

    def __init__(
        self,
        movies_path: Path | str = Path("data") / "movies.csv",
        tags_path: Path | str = Path("data") / "tags.csv",
        ratings_path: Path | str = Path("data") / "ratings.csv",
        alpha: float = 0.5,  # weight for content vs collaborative
    ) -> None:
        # Blend weight: alpha * content_score + (1 - alpha) * collaborative_score
        # Clamp to [0,1] to avoid misuse
        self.alpha = max(0.0, min(1.0, alpha))
        self.content = ContentRecommender(movies_path, tags_path)
        self.collab = CollaborativeRecommender(ratings_path)
        self._movies: Optional[pd.DataFrame] = None

    def fit(self) -> None:
        """Fit both underlying recommenders.

        Content is responsible for title ↔ metadata similarity; collaborative
        supplies user preference patterns. We cache the movies DataFrame for
        quick enrichment of results (titles & genres) without reloading.
        """
        self.content.fit()
        self.collab.fit()
        if self.content.is_fitted():  # Cache for enrichment
            self._movies = self.content._movies

    # -------------------- Title-based -------------------- #
    def recommend_by_title(self, title: str, n: int = 10) -> List[Dict]:
        """Recommend similar movies to a given title (pure content-based).

        Falls back to empty list if content model not fitted or title missing.
        """
        return self.content.recommend(title, n)

    # -------------------- User-based -------------------- #
    def recommend_by_user(self, user_id: int, n: int = 10) -> List[Dict]:
        """Collaborative recommendations for a user.

        Adds title/genres if movies metadata available.
        """
        recs = self.collab.recommend(user_id, n)
        if not recs:
            return []
        if self._movies is not None:
            df = pd.DataFrame(recs)
            merged = df.merge(self._movies, on="movieId", how="left")
            # Ensure columns exist even if metadata missing
            if "title" not in merged:
                merged["title"] = None
            if "genres" not in merged:
                merged["genres"] = None
            return merged[["movieId", "title", "genres", "score"]].to_dict("records")
        return recs  # Already contains movieId + score

    # -------------------- Hybrid blend -------------------- #
    def recommend_hybrid(self, user_id: int, title: str, n: int = 10) -> List[Dict]:
        """Blend content similarity (by title) and collaborative user preference.

        Steps:
          1. Ensure both models fitted (lazy fit).
          2. Retrieve top-2n candidates from each source (broad pool).
          3. Assign an inverse-rank score to content results (higher rank -> higher weight).
          4. Take raw latent score from collaborative model.
          5. Blend with alpha.
          6. Rank, truncate to n, enrich with metadata.
        """
        if not self.content.is_fitted() or not self.collab.is_fitted():
            self.fit()

        content_list = self.content.recommend(title, n * 2)
        collab_list = self.collab.recommend(user_id, n * 2)
        if not content_list and not collab_list:
            return []

        # Map movieId -> rank index for content (0 = best)
        content_ranks = {rec["movieId"]: rank for rank, rec in enumerate(content_list)}
        max_rank = max(content_ranks.values(), default=0)
        # Inverse-rank score (avoid div-by-zero; if single item -> score=1)
        content_scores = {
            mid: 1.0 if max_rank == 0 else 1 - (r / (max_rank + 1))
            for mid, r in content_ranks.items()
        }

        collab_scores = {rec["movieId"]: rec.get("score", 0.0) for rec in collab_list}

        # Aggregate union of candidate ids
        all_ids = set(content_scores) | set(collab_scores)
        blended = {}
        for mid in all_ids:
            c = content_scores.get(mid, 0.0)
            cf = collab_scores.get(mid, 0.0)
            blended[mid] = self.alpha * c + (1 - self.alpha) * cf

        # Rank blended scores
        top = sorted(blended.items(), key=lambda x: x[1], reverse=True)[:n]
        results: List[Dict] = []
        if self._movies is not None:
            for mid, sc in top:
                row = self._movies[self._movies.movieId == mid]
                if row.empty:
                    continue
                results.append({
                    "movieId": mid,
                    "title": row.iloc[0].title,
                    "genres": row.iloc[0].genres,
                    "score": round(float(sc), 4)
                })
        else:
            # Fallback without metadata
            results = [
                {"movieId": mid, "score": round(float(sc), 4)} for mid, sc in top
            ]
        return results


if __name__ == "__main__":
    hr = HybridRecommender()
    hr.fit()
    print("By Title:", hr.recommend_by_title("Toy Story (1995)", 5))
    print("By User:", hr.recommend_by_user(1, 5))
    print("Hybrid:", hr.recommend_hybrid(1, "Toy Story (1995)", 5))
