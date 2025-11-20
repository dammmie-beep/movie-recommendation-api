import argparse
import logging
from pathlib import Path
import mlflow
import mlflow.sklearn
import pandas as pd
from typing import Dict

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_loader import MovieLensLoader
from src.models.collaborative import CollaborativeFilteringModel, train_and_compare_models

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_mlflow(experiment_name: str = "movie-recommendation"):
    """Setup MLflow experiment tracking."""
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow experiment: {experiment_name}")
    logger.info("View experiments at: http://localhost:5000 (run: mlflow ui)")


def train_single_model(
    ratings_df: pd.DataFrame,
    algorithm: str,
    params: Dict = None
) -> tuple:
    """
    Train a single model with MLflow tracking.
    
    Args:
        ratings_df: Ratings DataFrame
        algorithm: Algorithm name
        params: Model parameters
        
    Returns:
        Tuple of (model, metrics)
    """
    with mlflow.start_run(run_name=f"{algorithm}_model"):
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {algorithm.upper()} model")
        logger.info(f"{'='*60}")
        
        # Log parameters
        if params is None:
            params = {}
        mlflow.log_params({
            "algorithm": algorithm,
            **params
        })
        
        # Log dataset info
        mlflow.log_params({
            "n_ratings": len(ratings_df),
            "n_users": ratings_df['userId'].nunique(),
            "n_movies": ratings_df['movieId'].nunique(),
        })
        
        # Train model
        model = CollaborativeFilteringModel(algorithm=algorithm, **params)
        metrics = model.train(ratings_df, test_size=0.2)
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Save model artifact
        model_path = f"data/models/{algorithm}_model.pkl"
        model.save(model_path)
        mlflow.log_artifact(model_path)
        
        logger.info(f"✅ {algorithm.upper()} complete - RMSE: {metrics['rmse']:.4f}")
        
        return model, metrics


def train_all_models(ratings_df: pd.DataFrame, quick_mode: bool = False):
    """
    Train all model variants and compare them.
    
    Args:
        ratings_df: Ratings DataFrame
        quick_mode: Use smaller parameters for faster training
    """
    logger.info("\n" + "="*60)
    logger.info("Training Multiple Models")
    logger.info("="*60)
    
    # Define models to train
    if quick_mode:
        logger.info("⚡ Quick mode enabled - using reduced parameters")
        models_config = {
            "svd": {"n_factors": 50, "n_epochs": 10},
            "knn_means": {"k": 20},
        }
    else:
        models_config = {
            "svd": {"n_factors": 100, "n_epochs": 20},
            "svdpp": {"n_factors": 20, "n_epochs": 10},
            "knn_basic": {"k": 40},
            "knn_means": {"k": 40},
        }
    
    results = {}
    
    for algorithm, params in models_config.items():
        try:
            model, metrics = train_single_model(ratings_df, algorithm, params)
            results[algorithm] = {
                'model': model,
                'metrics': metrics
            }
        except Exception as e:
            logger.error(f"❌ Failed to train {algorithm}: {e}")
            continue
    
    # Compare results
    logger.info("\n" + "="*60)
    logger.info("Model Comparison")
    logger.info("="*60)
    
    comparison_data = {
        algo: result['metrics']
        for algo, result in results.items()
    }
    
    comparison_df = pd.DataFrame(comparison_data).T
    comparison_df = comparison_df.sort_values('rmse')
    
    logger.info("\n" + comparison_df.to_string())
    
    # Select best model
    best_algo = comparison_df.index[0]
    best_metrics = results[best_algo]['metrics']
    best_model = results[best_algo]['model']
    
    logger.info(f"\n🏆 Best model: {best_algo.upper()}")
    logger.info(f"   RMSE: {best_metrics['rmse']:.4f}")
    logger.info(f"   MAE: {best_metrics['mae']:.4f}")
    
    # Save best model
    best_model_path = "data/models/best_model.pkl"
    best_model.save(best_model_path)
    logger.info(f"\n💾 Best model saved to: {best_model_path}")
    
    # Log best model to MLflow
    with mlflow.start_run(run_name="best_model"):
        mlflow.log_params({"algorithm": best_algo})
        mlflow.log_metrics(best_metrics)
        mlflow.log_artifact(best_model_path)
        mlflow.set_tag("best_model", "true")
    
    return results, best_algo


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train recommendation models")
    parser.add_argument(
        "--dataset",
        type=str,
        default="1m",
        choices=["1m", "25m"],
        help="Dataset size to use"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default=None,
        choices=["svd", "svdpp", "knn_basic", "knn_means"],
        help="Train single algorithm (default: train all)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick training mode with reduced parameters"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Sample N ratings for faster training"
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow tracking"
    )
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("Movie Recommendation System - Training")
    logger.info("="*60)
    
    # Setup MLflow
    if not args.no_mlflow:
        setup_mlflow()
    
    # Load data
    logger.info(f"\nLoading MovieLens {args.dataset.upper()} dataset...")
    data_path = f"data/raw/ml-{args.dataset}"
    
    if not Path(data_path).exists():
        logger.error(f"❌ Dataset not found at {data_path}")
        logger.error("Run: python scripts/download_data.py")
        return
    
    loader = MovieLensLoader(data_path)
    ratings, movies, users = loader.load_all()
    
    # Print statistics
    stats = loader.get_statistics()
    logger.info("\nDataset Statistics:")
    logger.info(f"  Ratings: {stats['n_ratings']:,}")
    logger.info(f"  Users: {stats['n_users']:,}")
    logger.info(f"  Movies: {stats['n_movies']:,}")
    logger.info(f"  Sparsity: {stats['sparsity']:.2%}")
    logger.info(f"  Avg rating: {stats['mean_rating']:.2f}")
    
    # Sample data if requested
    if args.sample:
        logger.info(f"\n⚠️  Sampling {args.sample:,} ratings for quick training")
        ratings = ratings.sample(n=args.sample, random_state=42)
    
    # Create models directory
    Path("data/models").mkdir(parents=True, exist_ok=True)
    
    # Train models
    if args.algorithm:
        # Train single model
        logger.info(f"\nTraining single model: {args.algorithm}")
        model, metrics = train_single_model(ratings, args.algorithm)
        
        # Also save as best model
        model.save("data/models/best_model.pkl")
        logger.info("✅ Model saved as best_model.pkl")
        
    else:
        # Train all models
        results, best_algo = train_all_models(ratings, quick_mode=args.quick)
    
    logger.info("\n" + "="*60)
    logger.info("Training Complete! 🎉")
    logger.info("="*60)
    logger.info("\nNext steps:")
    logger.info("1. View experiments: mlflow ui")
    logger.info("2. Start API: uvicorn src.api.main:app --reload")
    logger.info("3. Test API: http://localhost:8000/docs")
    logger.info("="*60)


if __name__ == "__main__":
    main()