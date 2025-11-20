"""
FastAPI application for movie recommendation system.
Provides REST API for getting recommendations and predictions.
"""

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi import Path as PathParam
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging
from pathlib import Path
import time
from functools import lru_cache
import pandas as pd

from src.models.collaborative import CollaborativeFilteringModel
from src.data.data_loader import MovieLensLoader

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Movie Recommendation API",
    description="Production-ready movie recommendation system using collaborative filtering",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and data
model = None
movies_df = None
ratings_df = None


# Pydantic models for request/response
class RecommendationResponse(BaseModel):
    """Response model for recommendations."""
    user_id: int
    recommendations: List[Dict[str, Any]] = Field(..., description="List of recommended movies")
    timestamp: str
    processing_time_ms: float


class PredictionRequest(BaseModel):
    """Request model for rating prediction."""
    user_id: int = Field(..., ge=1, description="User ID")
    movie_id: int = Field(..., ge=1, description="Movie ID")


class PredictionResponse(BaseModel):
    """Response model for rating prediction."""
    user_id: int
    movie_id: int
    predicted_rating: float
    movie_title: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_type: Optional[str] = None
    data_loaded: bool
    n_movies: Optional[int] = None
    n_ratings: Optional[int] = None


class MovieInfo(BaseModel):
    """Movie information."""
    movie_id: int
    title: str
    genres: str
    avg_rating: Optional[float] = None
    n_ratings: Optional[int] = None


# Dependency injection for model loading
@lru_cache()
def get_model() -> CollaborativeFilteringModel:
    """Load model (cached)."""
    model_path = Path("data/models/best_model.pkl")
    
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        raise HTTPException(status_code=503, detail="Model not available. Please train a model first.")
    
    logger.info("Loading model...")
    loaded_model = CollaborativeFilteringModel.load(str(model_path))
    logger.info("Model loaded successfully")
    
    return loaded_model


@lru_cache()
def get_data() -> tuple:
    """Load movie and ratings data (cached)."""
    logger.info("Loading data...")
    
    loader = MovieLensLoader("data/raw/ml-1m")
    ratings, movies, _ = loader.load_all()
    
    logger.info(f"Loaded {len(movies)} movies and {len(ratings)} ratings")
    
    return ratings, movies


# Startup event
@app.on_event("startup")
async def startup_event():
    """Load model and data on startup."""
    global model, movies_df, ratings_df
    
    try:
        model = get_model()
        ratings_df, movies_df = get_data()
        logger.info("✅ Application startup complete")
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        # Don't fail startup - let endpoints handle missing model


# Health check endpoint
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    Returns the status of the API and whether the model is loaded.
    """
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        model_type=model.algorithm_name if model else None,
        data_loaded=movies_df is not None,
        n_movies=len(movies_df) if movies_df is not None else None,
        n_ratings=len(ratings_df) if ratings_df is not None else None
    )


# Root endpoint
@app.get("/", tags=["Info"])
async def root():
    """API information."""
    return {
        "message": "Movie Recommendation API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


# Get recommendations
@app.get("/recommend/{user_id}", response_model=RecommendationResponse, tags=["Recommendations"])
async def get_recommendations(
    user_id: int = PathParam(..., ge=1, description="User ID to get recommendations for"),
    n: int = Query(10, ge=1, le=100, description="Number of recommendations"),
    include_details: bool = Query(True, description="Include movie details")
):
    """
    Get personalized movie recommendations for a user.
    
    Returns top-N movies the user is likely to enjoy based on collaborative filtering.
    """
    start_time = time.time()
    
    if model is None or movies_df is None:
        raise HTTPException(status_code=503, detail="Service not ready. Model or data not loaded.")
    
    try:
        # Get recommendations from model
        recommendations = model.recommend(user_id=user_id, n=n)
        
        # Format response
        results = []
        for movie_id, predicted_rating in recommendations:
            result = {
                "movie_id": int(movie_id),
                "predicted_rating": round(float(predicted_rating), 2)
            }
            
            if include_details and movies_df is not None:
                movie_info = movies_df[movies_df['movieId'] == movie_id]
                if not movie_info.empty:
                    result["title"] = movie_info.iloc[0]['title']
                    result["genres"] = movie_info.iloc[0]['genres']
                    
                    # Add average rating if available
                    if ratings_df is not None:
                        movie_ratings = ratings_df[ratings_df['movieId'] == movie_id]['rating']
                        if not movie_ratings.empty:
                            result["avg_rating"] = round(float(movie_ratings.mean()), 2)
                            result["n_ratings"] = int(len(movie_ratings))
            
            results.append(result)
        
        processing_time = (time.time() - start_time) * 1000
        
        return RecommendationResponse(
            user_id=user_id,
            recommendations=results,
            timestamp=pd.Timestamp.now().isoformat(),
            processing_time_ms=round(processing_time, 2)
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found: {str(e)}")
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# Predict rating
@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_rating(request: PredictionRequest):
    """
    Predict rating for a specific user-movie pair.
    
    Useful for:
    - A/B testing different recommendations
    - Explaining why a movie was recommended
    - Building custom recommendation logic
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        predicted_rating = model.predict(
            user_id=request.user_id,
            movie_id=request.movie_id
        )
        
        # Get movie title if available
        movie_title = None
        if movies_df is not None:
            movie_info = movies_df[movies_df['movieId'] == request.movie_id]
            if not movie_info.empty:
                movie_title = movie_info.iloc[0]['title']
        
        return PredictionResponse(
            user_id=request.user_id,
            movie_id=request.movie_id,
            predicted_rating=round(float(predicted_rating), 2),
            movie_title=movie_title
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Get movie details
@app.get("/movies/{movie_id}", response_model=MovieInfo, tags=["Movies"])
async def get_movie_info(movie_id: int):
    """
    Get detailed information about a specific movie.
    """
    if movies_df is None:
        raise HTTPException(status_code=503, detail="Movie data not loaded")
    
    movie = movies_df[movies_df['movieId'] == movie_id]
    
    if movie.empty:
        raise HTTPException(status_code=404, detail=f"Movie {movie_id} not found")
    
    movie_data = movie.iloc[0]
    
    # Calculate statistics from ratings
    avg_rating = None
    n_ratings = None
    if ratings_df is not None:
        movie_ratings = ratings_df[ratings_df['movieId'] == movie_id]['rating']
        if not movie_ratings.empty:
            avg_rating = round(float(movie_ratings.mean()), 2)
            n_ratings = int(len(movie_ratings))
    
    return MovieInfo(
        movie_id=int(movie_data['movieId']),
        title=movie_data['title'],
        genres=movie_data['genres'],
        avg_rating=avg_rating,
        n_ratings=n_ratings
    )


# Search movies
@app.get("/movies/search/", response_model=List[MovieInfo], tags=["Movies"])
async def search_movies(
    query: str = Query(..., min_length=2, description="Search query"),
    limit: int = Query(10, ge=1, le=50, description="Max results")
):
    """
    Search for movies by title.
    """
    if movies_df is None:
        raise HTTPException(status_code=503, detail="Movie data not loaded")
    
    # Case-insensitive search
    mask = movies_df['title'].str.contains(query, case=False, na=False)
    results = movies_df[mask].head(limit)
    
    if results.empty:
        return []
    
    # Format results
    movie_list = []
    for _, movie in results.iterrows():
        # Get rating stats
        avg_rating = None
        n_ratings = None
        if ratings_df is not None:
            movie_ratings = ratings_df[ratings_df['movieId'] == movie['movieId']]['rating']
            if not movie_ratings.empty:
                avg_rating = round(float(movie_ratings.mean()), 2)
                n_ratings = int(len(movie_ratings))
        
        movie_list.append(MovieInfo(
            movie_id=int(movie['movieId']),
            title=movie['title'],
            genres=movie['genres'],
            avg_rating=avg_rating,
            n_ratings=n_ratings
        ))
    
    return movie_list


# Get popular movies
@app.get("/movies/popular/", response_model=List[MovieInfo], tags=["Movies"])
async def get_popular_movies(
    limit: int = Query(20, ge=1, le=100),
    min_ratings: int = Query(100, ge=10, description="Minimum number of ratings")
):
    """
    Get most popular movies based on number of ratings.
    """
    if movies_df is None or ratings_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    # Calculate movie statistics
    movie_stats = ratings_df.groupby('movieId').agg({
        'rating': ['mean', 'count']
    }).reset_index()
    
    movie_stats.columns = ['movieId', 'avg_rating', 'n_ratings']
    
    # Filter by minimum ratings
    popular = movie_stats[movie_stats['n_ratings'] >= min_ratings]
    popular = popular.sort_values('n_ratings', ascending=False).head(limit)
    
    # Merge with movie info
    popular = popular.merge(movies_df, on='movieId')
    
    # Format response
    result = []
    for _, movie in popular.iterrows():
        result.append(MovieInfo(
            movie_id=int(movie['movieId']),
            title=movie['title'],
            genres=movie['genres'],
            avg_rating=round(float(movie['avg_rating']), 2),
            n_ratings=int(movie['n_ratings'])
        ))
    
    return result


# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Global error handler caught: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again later."}
    )


if __name__ == "__main__":
    import uvicorn
    
    # Run the API
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )