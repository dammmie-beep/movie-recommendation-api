"""
Data loading utilities for MovieLens dataset.
Handles both 1M and 25M versions.
"""

import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MovieLensLoader:
    """Load and prepare MovieLens data."""
    
    def __init__(self, data_path: str = "data/raw/ml-1m"):
        """
        Initialize loader.
        
        Args:
            data_path: Path to MovieLens dataset directory
        """
        self.data_path = Path(data_path)
        self.ratings = None
        self.movies = None
        self.users = None
        
    def load_ratings(self) -> pd.DataFrame:
        """
        Load ratings data.
        
        Returns:
            DataFrame with columns: userId, movieId, rating, timestamp
        """
        ratings_file = self.data_path / "ratings.dat"
        
        if not ratings_file.exists():
            # Try alternative naming (25M dataset)
            ratings_file = self.data_path / "ratings.csv"
            
        if ratings_file.suffix == ".dat":
            # 1M dataset format
            self.ratings = pd.read_csv(
                ratings_file,
                sep="::",
                engine="python",
                names=["userId", "movieId", "rating", "timestamp"],
                encoding="latin-1"
            )
        else:
            # 25M dataset format (CSV)
            self.ratings = pd.read_csv(ratings_file)
            
        logger.info(f"Loaded {len(self.ratings):,} ratings")
        logger.info(f"Users: {self.ratings['userId'].nunique():,}")
        logger.info(f"Movies: {self.ratings['movieId'].nunique():,}")
        logger.info(f"Sparsity: {self._calculate_sparsity():.2%}")
        
        return self.ratings
    
    def load_movies(self) -> pd.DataFrame:
        """
        Load movies metadata.
        
        Returns:
            DataFrame with columns: movieId, title, genres
        """
        movies_file = self.data_path / "movies.dat"
        
        if not movies_file.exists():
            movies_file = self.data_path / "movies.csv"
            
        if movies_file.suffix == ".dat":
            self.movies = pd.read_csv(
                movies_file,
                sep="::",
                engine="python",
                names=["movieId", "title", "genres"],
                encoding="latin-1"
            )
        else:
            self.movies = pd.read_csv(movies_file)
            
        logger.info(f"Loaded {len(self.movies):,} movies")
        
        return self.movies
    
    def load_users(self) -> Optional[pd.DataFrame]:
        """
        Load user demographics (only available in 1M dataset).
        
        Returns:
            DataFrame with user information or None
        """
        users_file = self.data_path / "users.dat"
        
        if not users_file.exists():
            logger.warning("Users file not found (only in 1M dataset)")
            return None
            
        self.users = pd.read_csv(
            users_file,
            sep="::",
            engine="python",
            names=["userId", "gender", "age", "occupation", "zip"],
            encoding="latin-1"
        )
        
        logger.info(f"Loaded {len(self.users):,} user profiles")
        
        return self.users
    
    def load_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Load all datasets.
        
        Returns:
            Tuple of (ratings, movies, users)
        """
        ratings = self.load_ratings()
        movies = self.load_movies()
        users = self.load_users()
        
        return ratings, movies, users
    
    def get_merged_data(self) -> pd.DataFrame:
        """
        Get ratings merged with movie information.
        
        Returns:
            DataFrame with ratings and movie metadata
        """
        if self.ratings is None:
            self.load_ratings()
        if self.movies is None:
            self.load_movies()
            
        merged = self.ratings.merge(self.movies, on="movieId", how="left")
        
        return merged
    
    def _calculate_sparsity(self) -> float:
        """Calculate rating matrix sparsity."""
        if self.ratings is None:
            return 0.0
            
        n_users = self.ratings['userId'].nunique()
        n_movies = self.ratings['movieId'].nunique()
        n_ratings = len(self.ratings)
        
        sparsity = 1 - (n_ratings / (n_users * n_movies))
        return sparsity
    
    def get_statistics(self) -> dict:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with dataset stats
        """
        if self.ratings is None:
            self.load_ratings()
            
        stats = {
            "n_ratings": len(self.ratings),
            "n_users": self.ratings['userId'].nunique(),
            "n_movies": self.ratings['movieId'].nunique(),
            "sparsity": self._calculate_sparsity(),
            "min_rating": self.ratings['rating'].min(),
            "max_rating": self.ratings['rating'].max(),
            "mean_rating": self.ratings['rating'].mean(),
            "median_rating": self.ratings['rating'].median(),
            "ratings_per_user": self.ratings.groupby('userId').size().describe().to_dict(),
            "ratings_per_movie": self.ratings.groupby('movieId').size().describe().to_dict(),
        }
        
        return stats


def quick_load(dataset_size: str = "1m") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Quick load function for convenience.
    
    Args:
        dataset_size: "1m" or "25m"
    
    Returns:
        Tuple of (ratings, movies)
    """
    data_path = f"data/raw/ml-{dataset_size}"
    loader = MovieLensLoader(data_path)
    ratings, movies, _ = loader.load_all()
    
    return ratings, movies


if __name__ == "__main__":
    # Test the loader
    print("Testing MovieLens Loader...")
    print("=" * 60)
    
    loader = MovieLensLoader()
    ratings, movies, users = loader.load_all()
    
    print("\nDataset Statistics:")
    print("-" * 60)
    stats = loader.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        elif isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v:.2f}")
        else:
            print(f"{key}: {value:,}")
    
    print("\nSample data:")
    print("-" * 60)
    print("\nRatings:")
    print(ratings.head())
    print("\nMovies:")
    print(movies.head())
    
    if users is not None:
        print("\nUsers:")
        print(users.head())