"""
Collaborative filtering models using Surprise library.
Implements SVD, SVD++, and KNN-based approaches.
"""

import pickle
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from surprise import SVD, SVDpp, KNNBasic, KNNWithMeans, Dataset, Reader
from surprise.model_selection import train_test_split, cross_validate
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CollaborativeFilteringModel:
    """Wrapper for Surprise collaborative filtering models."""
    
    def __init__(self, algorithm: str = "svd", **kwargs):
        """
        Initialize model.
        
        Args:
            algorithm: One of ["svd", "svdpp", "knn_basic", "knn_means"]
            **kwargs: Algorithm-specific parameters
        """
        self.algorithm_name = algorithm
        self.model = self._initialize_algorithm(algorithm, **kwargs)
        self.is_trained = False
        self.trainset = None
        
    def _initialize_algorithm(self, algorithm: str, **kwargs):
        """Initialize the specified algorithm."""
        algorithms = {
            "svd": SVD,
            "svdpp": SVDpp,
            "knn_basic": KNNBasic,
            "knn_means": KNNWithMeans,
        }
        
        if algorithm not in algorithms:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Set default parameters
        if algorithm == "svd":
            default_params = {
                "n_factors": 100,
                "n_epochs": 20,
                "lr_all": 0.005,
                "reg_all": 0.02,
                "random_state": 42
            }
        elif algorithm == "svdpp":
            default_params = {
                "n_factors": 20,  # Fewer factors for SVD++ (slower)
                "n_epochs": 10,
                "random_state": 42
            }
        else:  # KNN methods
            default_params = {
                "k": 40,
                "sim_options": {
                    "name": "cosine",
                    "user_based": False  # Item-based by default
                }
            }
        
        # Override with user parameters
        default_params.update(kwargs)
        
        return algorithms[algorithm](**default_params)
    
    def train(self, ratings_df: pd.DataFrame, test_size: float = 0.2) -> Dict:
        """
        Train the model.
        
        Args:
            ratings_df: DataFrame with columns [userId, movieId, rating]
            test_size: Fraction of data to use for testing
            
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training {self.algorithm_name} model...")
        
        # Convert to Surprise format
        reader = Reader(rating_scale=(ratings_df['rating'].min(), 
                                      ratings_df['rating'].max()))
        data = Dataset.load_from_df(
            ratings_df[['userId', 'movieId', 'rating']], 
            reader
        )
        
        # Split data
        trainset, testset = train_test_split(data, test_size=test_size, random_state=42)
        
        # Train
        self.model.fit(trainset)
        self.trainset = trainset
        self.is_trained = True
        
        # Evaluate
        predictions = self.model.test(testset)
        metrics = self._calculate_metrics(predictions)
        
        logger.info(f"Training complete. RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}")
        
        return metrics
    
    def cross_validate(self, ratings_df: pd.DataFrame, cv: int = 5) -> Dict:
        """
        Perform cross-validation.
        
        Args:
            ratings_df: DataFrame with ratings
            cv: Number of folds
            
        Returns:
            Dictionary with CV metrics
        """
        logger.info(f"Performing {cv}-fold cross-validation...")
        
        reader = Reader(rating_scale=(ratings_df['rating'].min(), 
                                      ratings_df['rating'].max()))
        data = Dataset.load_from_df(
            ratings_df[['userId', 'movieId', 'rating']], 
            reader
        )
        
        results = cross_validate(
            self.model, 
            data, 
            measures=['RMSE', 'MAE'], 
            cv=cv, 
            verbose=False
        )
        
        cv_metrics = {
            'rmse_mean': results['test_rmse'].mean(),
            'rmse_std': results['test_rmse'].std(),
            'mae_mean': results['test_mae'].mean(),
            'mae_std': results['test_mae'].std(),
            'fit_time_mean': results['fit_time'].mean(),
            'test_time_mean': results['test_time'].mean()
        }
        
        logger.info(f"CV RMSE: {cv_metrics['rmse_mean']:.4f} (+/- {cv_metrics['rmse_std']:.4f})")
        
        return cv_metrics
    
    def predict(self, user_id: int, movie_id: int) -> float:
        """
        Predict rating for a user-movie pair.
        
        Args:
            user_id: User ID
            movie_id: Movie ID
            
        Returns:
            Predicted rating
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        prediction = self.model.predict(user_id, movie_id)
        return prediction.est
    
    def recommend(self, user_id: int, n: int = 10, 
                  candidate_movies: List[int] = None) -> List[Tuple[int, float]]:
        """
        Get top-N recommendations for a user.
        
        Args:
            user_id: User ID
            n: Number of recommendations
            candidate_movies: List of movie IDs to consider (if None, use all)
            
        Returns:
            List of (movie_id, predicted_rating) tuples
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before recommendation")
        
        # Get all movies if candidates not specified
        if candidate_movies is None:
            all_movies = self.trainset.all_items()
            # Convert inner IDs to raw IDs
            candidate_movies = [self.trainset.to_raw_iid(iid) for iid in all_movies]
        
        # Get user's already rated movies
        try:
            user_inner_id = self.trainset.to_inner_uid(user_id)
            rated_movies = set(
                self.trainset.to_raw_iid(iid) 
                for (iid, _) in self.trainset.ur[user_inner_id]
            )
        except ValueError:
            # New user - no rated movies
            rated_movies = set()
        
        # Filter out already rated movies
        candidate_movies = [mid for mid in candidate_movies if mid not in rated_movies]
        
        # Predict ratings for all candidates
        predictions = []
        for movie_id in candidate_movies:
            pred = self.model.predict(user_id, movie_id)
            predictions.append((movie_id, pred.est))
        
        # Sort by predicted rating and return top N
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions[:n]
    
    def get_similar_items(self, movie_id: int, n: int = 10) -> List[Tuple[int, float]]:
        """
        Get similar items (for KNN-based models).
        
        Args:
            movie_id: Movie ID
            n: Number of similar items
            
        Returns:
            List of (movie_id, similarity) tuples
        """
        if not hasattr(self.model, 'sim'):
            raise ValueError("Similarity matrix not available for this algorithm")
        
        try:
            inner_id = self.trainset.to_inner_iid(movie_id)
        except ValueError:
            logger.warning(f"Movie {movie_id} not in training set")
            return []
        
        # Get similarity scores
        similarities = self.model.sim[inner_id]
        
        # Get top N (excluding the item itself)
        similar_indices = np.argsort(similarities)[::-1][1:n+1]
        
        similar_items = [
            (self.trainset.to_raw_iid(idx), similarities[idx])
            for idx in similar_indices
        ]
        
        return similar_items
    
    def _calculate_metrics(self, predictions) -> Dict:
        """Calculate evaluation metrics."""
        # RMSE
        rmse = np.sqrt(np.mean([(pred.est - pred.r_ui)**2 for pred in predictions]))
        
        # MAE
        mae = np.mean([abs(pred.est - pred.r_ui) for pred in predictions])
        
        return {
            'rmse': rmse,
            'mae': mae,
            'n_predictions': len(predictions)
        }
    
    def save(self, filepath: str):
        """Save model to disk."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'algorithm_name': self.algorithm_name,
                'trainset': self.trainset
            }, f)
        
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Load model from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls.__new__(cls)
        instance.model = data['model']
        instance.algorithm_name = data['algorithm_name']
        instance.trainset = data['trainset']
        instance.is_trained = True
        
        logger.info(f"Model loaded from {filepath}")
        
        return instance


def train_and_compare_models(ratings_df: pd.DataFrame) -> Dict:
    """
    Train and compare multiple collaborative filtering algorithms.
    
    Args:
        ratings_df: DataFrame with ratings
        
    Returns:
        Dictionary with results for each algorithm
    """
    algorithms = ["svd", "svdpp", "knn_basic", "knn_means"]
    results = {}
    
    for algo in algorithms:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {algo.upper()}")
        logger.info(f"{'='*60}")
        
        model = CollaborativeFilteringModel(algorithm=algo)
        metrics = model.train(ratings_df)
        results[algo] = {
            'model': model,
            'metrics': metrics
        }
    
    # Print comparison
    logger.info("\n" + "="*60)
    logger.info("Model Comparison")
    logger.info("="*60)
    
    comparison_df = pd.DataFrame({
        algo: result['metrics'] 
        for algo, result in results.items()
    }).T
    
    logger.info("\n" + comparison_df.to_string())
    
    # Find best model
    best_algo = min(results.keys(), key=lambda x: results[x]['metrics']['rmse'])
    logger.info(f"\n🏆 Best model: {best_algo.upper()} (RMSE: {results[best_algo]['metrics']['rmse']:.4f})")
    
    return results


if __name__ == "__main__":
    # Test with sample data
    from src.data.loader import quick_load
    
    print("Loading data...")
    ratings, _ = quick_load("1m")
    
    # Use subset for quick testing
    ratings_sample = ratings.sample(n=50000, random_state=42)
    
    print("\nTraining SVD model...")
    model = CollaborativeFilteringModel(algorithm="svd")
    metrics = model.train(ratings_sample)
    
    print("\nGetting recommendations for user 1...")
    recommendations = model.recommend(user_id=1, n=10)
    
    print("\nTop 10 recommendations:")
    for i, (movie_id, rating) in enumerate(recommendations, 1):
        print(f"{i}. Movie {movie_id}: {rating:.2f}")
    
    print("\nSaving model...")
    model.save("data/models/svd_model.pkl")