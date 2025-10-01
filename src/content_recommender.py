"""Content-based recommender implementation.

This module defines a lightweight `ContentRecommender` class used by the
hybrid recommender. It safely loads data (if available), builds a TF-IDF
matrix over textual features, and supports title-based recommendations.

Important notes:
 - CSV files are ignored by git; ensure you have placed `movies.csv` and
   `tags.csv` inside the `data/` directory (or pass custom paths).
 - The class will no-op (and return empty recommendations) if data files
   are missing, instead of raising import-time errors.
 - The previous version executed code at import time and contained bugs:
     * Used non-existent column name `movies['tags']` (should be `tag`).
     * Returned DataFrame.tolist() which produced column lists.
     * Printed recommendations on import (side effects removed).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Union

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentRecommender:
    """Content-based movie recommender using TF-IDF over title/genres/tags."""

    def __init__(
        self,
        movies_path: Union[str, Path] = Path("data") / "movies.csv",
        tags_path: Union[str, Path] = Path("data") / "tags.csv",
    ) -> None:
        self.movies_path = Path(movies_path)
        self.tags_path = Path(tags_path)
        self._movies: pd.DataFrame | None = None
        self._tfidf = None
        self._tfidf_matrix = None
        self._cosine_sim = None
        self._indices: pd.Series | None = None
        self._fitted: bool = False

    def fit(self) -> None:
        """Load data and build similarity matrix.

        Safe to call multiple times; rebuilds internal structures.
        Missing files result in a silent no-op (fitted remains False).
        """
        # Guard: if either data file is missing we simply mark as not fitted
        if not self.movies_path.exists() or not self.tags_path.exists():
            # Silently skip if data not available
            self._fitted = False
            return

        # Load datasets
        movies = pd.read_csv(self.movies_path)
        tags = pd.read_csv(self.tags_path)

        # Defensive cleaning
        if 'movieId' not in movies.columns:
            raise ValueError("'movies.csv' must contain a 'movieId' column")
        if 'title' not in movies.columns:
            raise ValueError("'movies.csv' must contain a 'title' column")
        if 'genres' not in movies.columns:
            # Some datasets may omit genres; fill with empty string
            movies['genres'] = ""

        if 'movieId' not in tags.columns or 'tag' not in tags.columns:
            # Create empty tags frame if structure unexpected
            tags = pd.DataFrame({'movieId': [], 'tag': []})

        # Normalise tag text & aggregate all tags per movie into one string
        tags['tag'] = tags['tag'].fillna("")
        agg_tags = (
            tags.groupby('movieId')['tag']
            .apply(lambda x: " ".join(str(t) for t in x if t))
            .reset_index()
        )

        # Left join so movies without tags are retained
        movies = movies.merge(agg_tags, on='movieId', how='left')
        movies['tag'] = movies['tag'].fillna("")

        # Build combined textual feature
        movies['content'] = (
            movies['title'].fillna("") + " " +
            movies['genres'].fillna("") + " " +
            movies['tag'].fillna("")
        )

        # Vectorize
        # TF-IDF converts the free-text (title + genres + tags) into a sparse
        # numeric matrix where each column is a term-weight. Stop words removed.
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(movies['content'])
        # Do NOT compute the full NxN cosine similarity matrix here — it's
        # memory-prohibitive for large datasets. We'll compute similarities
        # on-demand in `recommend` by comparing a single movie vector to the
        # TF-IDF matrix.

        # Store artifacts
        # Reset index so we have a clean 0..N-1 mapping for row lookups
        movies = movies.reset_index(drop=True)
        # Map from title -> row index (drop duplicates to first occurrence)
        indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

        self._movies = movies
        self._tfidf = tfidf
        self._tfidf_matrix = tfidf_matrix
        self._indices = indices
        self._fitted = True

    def is_fitted(self) -> bool:
        return self._fitted

    def recommend(self, title: str, n: int = 10) -> List[dict]:
        """Recommend similar movies by title.

        Args:
            title: Exact movie title (case-sensitive match preferred).
            n: Number of similar movies to return.

        Returns:
            A list of dicts with keys: movieId, title, genres
        """
        if not self._fitted:
            # Attempt lazy fit
            self.fit()
            if not self._fitted:
                # Still not fitted (likely missing data files) -> empty result
                return []

        assert self._movies is not None
        assert self._tfidf_matrix is not None
        assert self._indices is not None

        if title not in self._indices:
            # Try case-insensitive fallback
            lower_map = {t.lower(): t for t in self._indices.index}
            if title.lower() in lower_map:
                title = lower_map[title.lower()]
            else:
                # Title genuinely not found
                return []

        idx = self._indices[title]

        # Compute similarity between the target movie vector and all movies.
        # The result is a (1, N) dense array which is small compared to an
        # N x N matrix.
        target_vec = self._tfidf_matrix[idx]
        sim_array = cosine_similarity(target_vec, self._tfidf_matrix).flatten()

        # Get top-n indices (exclude the movie itself)
        sim_indices = sim_array.argsort()[::-1]
        sim_indices = [i for i in sim_indices if i != idx][:n]
        movie_indices = sim_indices
        # Slice only required columns and convert each row to a dict
        result_df = self._movies.loc[movie_indices, ['movieId', 'title', 'genres']]
        return result_df.to_dict(orient='records')


# Convenience function mirroring previous API (optional)
_default_recommender: ContentRecommender | None = None


def recommend(title: str, n: int = 10):  # pragma: no cover - thin wrapper
    global _default_recommender
    if _default_recommender is None:
        _default_recommender = ContentRecommender()
        _default_recommender.fit()
    return _default_recommender.recommend(title, n)


if __name__ == "__main__":  # Manual quick test (will silently return [])
    sample_titles = ["Toy Story (1995)", "The Matrix (1999)"]
    for t in sample_titles:
        recs = recommend(t, 5)
        print(f"Recommendations for {t}: {len(recs)} found")
