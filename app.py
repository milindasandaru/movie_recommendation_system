"""Command-line interface for the Movie Recommendation System.

Provides three main modes:
 1. content  - Recommend similar movies to a given title
 2. user     - Recommend movies for a user via collaborative filtering
 3. hybrid   - Blend content + collaborative (requires both title and user)

Usage examples:
  python app.py content --title "Toy Story (1995)" --top 5
  python app.py user --user 1 --top 10
  python app.py hybrid --user 1 --title "Toy Story (1995)" --alpha 0.6 --top 8

Make sure CSV data files are placed under the ./data directory as per data/README.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.content_recommender import ContentRecommender
from src.collaborative_recommender import CollaborativeRecommender
from src.hybrid_recommender import HybridRecommender


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Movie Recommendation System CLI",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	sub = parser.add_subparsers(dest="mode", required=True)

	# Content
	p_content = sub.add_parser("content", help="Content-based similar titles")
	p_content.add_argument("--title", required=True, help="Exact movie title")
	p_content.add_argument("--top", type=int, default=10, help="Number of results")

	# User (collaborative)
	p_user = sub.add_parser("user", help="Collaborative user recommendations")
	p_user.add_argument("--user", type=int, required=True, help="User ID")
	p_user.add_argument("--top", type=int, default=10, help="Number of results")

	# Hybrid
	p_hybrid = sub.add_parser("hybrid", help="Hybrid (content + collaborative)")
	p_hybrid.add_argument("--user", type=int, required=True, help="User ID")
	p_hybrid.add_argument("--title", required=True, help="Anchor movie title")
	p_hybrid.add_argument("--alpha", type=float, default=0.5, help="Blend weight (content)")
	p_hybrid.add_argument("--top", type=int, default=10, help="Number of results")

	parser.add_argument("--data-dir", default="data", help="Data directory path")
	parser.add_argument("--json", action="store_true", help="Output JSON only")
	return parser


def ensure_data_files(data_dir: Path) -> dict[str, bool]:
	required = {
		"movies.csv": (data_dir / "movies.csv").exists(),
		"ratings.csv": (data_dir / "ratings.csv").exists(),
		"tags.csv": (data_dir / "tags.csv").exists(),  # optional but useful
	}
	return required


def main(argv: list[str] | None = None) -> int:
	parser = build_parser()
	args = parser.parse_args(argv)
	data_dir = Path(args.data_dir)

	availability = ensure_data_files(data_dir)
	missing_core = not availability["movies.csv"] or not availability["ratings.csv"]
	if missing_core and args.mode in {"user", "hybrid"}:
		parser.error("movies.csv and ratings.csv are required for this mode.")

	# Instantiate models (lazy fitting inside their methods when possible)
	content = ContentRecommender(movies_path=data_dir / "movies.csv", tags_path=data_dir / "tags.csv")
	collab = CollaborativeRecommender(ratings_path=data_dir / "ratings.csv")

	if args.mode == "content":
		content.fit()
		recs = content.recommend(args.title, args.top)
		output(recs, args.json)
		return 0

	if args.mode == "user":
		collab.fit()
		recs = collab.recommend(args.user, args.top)
		output(recs, args.json)
		return 0

	if args.mode == "hybrid":
		hr = HybridRecommender(
			movies_path=data_dir / "movies.csv",
			tags_path=data_dir / "tags.csv",
			ratings_path=data_dir / "ratings.csv",
			alpha=args.alpha,
		)
		hr.fit()
		recs = hr.recommend_hybrid(args.user, args.title, args.top)
		output(recs, args.json)
		return 0

	parser.error("Unknown mode")
	return 1


def output(recs: list[dict[str, Any]], as_json: bool) -> None:
	if as_json:
		print(json.dumps(recs, indent=2))
		return
	if not recs:
		print("No recommendations found.")
		return
	# Pretty table-like output
	headers = recs[0].keys()
	print(" | ".join(str(h) for h in headers))
	print("-" * 80)
	for r in recs:
		print(" | ".join(str(r.get(h, "")) for h in headers))


if __name__ == "__main__":  # pragma: no cover
	raise SystemExit(main())
