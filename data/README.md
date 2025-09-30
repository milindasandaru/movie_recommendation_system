# Movie Recommendation System - Data

This folder contains the datasets required for the movie recommendation system. Due to file size limitations, the CSV files are not included in the repository.

## Required Datasets

Please download the following datasets and place them in this folder:

### MovieLens Dataset
Download from: [MovieLens 25M Dataset](https://grouplens.org/datasets/movielens/25m/)

**Required files:**
- `movies.csv` - Movie information with movieId, title, and genres
- `ratings.csv` - User ratings with userId, movieId, rating, and timestamp  
- `tags.csv` - User-generated tags for movies
- `links.csv` - Links to other movie databases (IMDb, TMDb)
- `genome-scores.csv` - Movie-tag relevance scores
- `genome-tags.csv` - Tag descriptions

## Dataset Description

- **Size**: ~265MB (unzipped)
- **Movies**: ~62,000 movies
- **Ratings**: ~25 million ratings
- **Users**: ~162,000 users
- **Time period**: 1995-2019

## Download Instructions

1. Visit the MovieLens website: https://grouplens.org/datasets/movielens/25m/
2. Download the `ml-25m.zip` file
3. Extract the ZIP file
4. Copy the CSV files to this `data/` folder
5. The folder structure should look like:
   ```
   data/
   ├── README.md (this file)
   ├── movies.csv
   ├── ratings.csv
   ├── tags.csv
   ├── links.csv
   ├── genome-scores.csv
   └── genome-tags.csv
   ```

## Data Format

### movies.csv
- movieId: Unique movie identifier
- title: Movie title with release year
- genres: Pipe-separated list of genres

### ratings.csv  
- userId: Unique user identifier
- movieId: Movie identifier
- rating: Rating on 5-star scale (0.5-5.0)
- timestamp: Unix timestamp

### tags.csv
- userId: User identifier
- movieId: Movie identifier  
- tag: User-generated tag
- timestamp: Unix timestamp

## Citation

If you use this dataset, please cite:

F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19.

## License

The MovieLens datasets are made available under the terms of the Creative Commons Attribution 4.0 International License.