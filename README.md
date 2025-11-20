#  Movie Recommendation API

A production-ready movie recommendation system using collaborative filtering. Built with FastAPI, Surprise, and MLflow.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [API Endpoints](#api-endpoints)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- 

---

## 🎯 Overview

This project implements a movie recommendation system using collaborative filtering techniques. It provides a REST API that serves personalized movie recommendations based on user ratings from the MovieLens 1M dataset.

**Key Highlights:**
- Fast API with <100ms response time
- SVD-based collaborative filtering
- MLflow experiment tracking
- Docker-ready for deployment
- Interactive API documentation
- Trained on 1M real movie ratings

---

## ✨ Features

### Core Functionality
- **Personalised Recommendations** - Get top-N movie suggestions for any user
- **Movie Search** - Find movies by title
- **Popular Movies** - Discover trending films
- **Rating Prediction** - Predict ratings for specific user-movie pairs
- **Movie Details** - Get comprehensive movie information

### Technical Features
- **Multiple Algorithms** - SVD, SVD++, KNN-based filtering
- **Model Versioning** - MLflow for experiment tracking
- **Auto-Generated Docs** - Interactive Swagger/ReDoc documentation
- **Input Validation** - Pydantic models for type safety
- **Error Handling** - Comprehensive error responses
- **Health Checks** - API status monitoring

---

## Tech Stack

| Category | Technology |
|----------|-----------|
| **Language** | Python 3.8+ |
| **Web Framework** | FastAPI |
| **ML Library** | Scikit-Surprise |
| **Data Processing** | Pandas, NumPy |
| **Experiment Tracking** | MLflow |
| **API Server** | Uvicorn |
| **Validation** | Pydantic |
| **Testing** | Pytest |
| **Containerization** | Docker |

---

## Quick Start

### Prerequisites

- Python 3.8 or higher
- 2GB free disk space
- Internet connection (for dataset download)

### Installation

```bash
# Clone the repository
git clone [https://github.com/yourusername/movie-recommendation-api.git](https://github.com/dammmie-beep/movie-recommendation-api.git)
cd movie-recommendation-api

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download dataset (1M ratings, ~6MB)
python scripts/download_data.py --size 1m

# Train model (~3 minutes)
python src/train.py --quick --sample 50000

# Start API server
uvicorn src.api.main:app --reload
```

### Verify Installation

```bash
# Check API health
curl http://localhost:8000/health

# Get recommendations
curl "http://localhost:8000/recommend/1?n=5"
```

Visit **http://localhost:8000/docs** for interactive API documentation.

---

## API Endpoints

### Recommendations

**GET** `/recommend/{user_id}`

Get personalized movie recommendations for a user.

```bash
curl "http://localhost:8000/recommend/1?n=10&include_details=true"
```

**Response:**
```json
{
  "user_id": 1,
  "recommendations": [
    {
      "movie_id": 318,
      "predicted_rating": 4.87,
      "title": "Shawshank Redemption, The (1994)",
      "genres": "Crime|Drama",
      "avg_rating": 4.49,
      "n_ratings": 63366
    }
  ],
  "timestamp": "2025-11-17T19:45:00",
  "processing_time_ms": 45.2
}
```

### Movie Search

**GET** `/movies/search/`

Search for movies by title.

```bash
curl "http://localhost:8000/movies/search/?query=matrix&limit=5"
```

### Popular Movies

**GET** `/movies/popular/`

Get most popular movies based on rating count.

```bash
curl "http://localhost:8000/movies/popular/?limit=20&min_ratings=100"
```

### Rating Prediction

**POST** `/predict`

Predict rating for a specific user-movie pair.

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "movie_id": 318}'
```

### Movie Details

**GET** `/movies/{movie_id}`

Get detailed information about a specific movie.

```bash
curl "http://localhost:8000/movies/318"
```

### Health Check

**GET** `/health`

Check API status and loaded resources.

```bash
curl "http://localhost:8000/health"
```

---

## Model Performance

### Training Results

| Model | RMSE | MAE | Training Time |
|-------|------|-----|---------------|
| **SVD** | **0.98** | **0.79** | 2m 15s |
| KNN (Item) | 1.08 | 0.84 | 45s |

*Metrics on 50K sample with quick training mode*

### API Performance

- **Latency (p95)**: 95ms (first request), 5ms (cached)
- **Throughput**: 120+ requests/second
- **Model Size**: ~15MB
- **Memory Usage**: ~200MB

### Dataset Statistics

- **Ratings**: 1,000,209
- **Users**: 6,040
- **Movies**: 3,706
- **Sparsity**: 95.53%
- **Rating Scale**: 1-5 stars
- **Average Rating**: 3.58

---

## Project Structure

```
movie-recommendation-api/
├── data/
│   ├── raw/                    # MovieLens dataset
│   └── models/                 # Trained models
│       └── best_model.pkl
│
├── src/
│   ├── data/
│   │   └── loader.py          # Data loading utilities
│   ├── models/
│   │   └── collaborative.py   # ML models
│   ├── api/
│   │   └── main.py           # FastAPI application
│   ├── train.py              # Training script
│   └── config.py             # Configuration
│
├── scripts/
│   └── download_data.py      # Dataset downloader
│
├── tests/                     # Unit & integration tests
├── notebooks/                 # Jupyter notebooks (EDA)
├── docker/                    # Docker configuration
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── mlruns/                    # MLflow experiments
├── requirements.txt
├── README.md
└── .gitignore
```

---




---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_api.py -v
```

---

## Docker Deployment

### Build and Run

```bash
# Build image
docker build -t movie-rec-api:latest -f docker/Dockerfile .

# Run container
docker run -p 8000:8000 movie-rec-api:latest

# Or use docker-compose
docker-compose up
```

### Environment Variables

```env
MODEL_PATH=data/models/best_model.pkl
DATASET_PATH=data/raw/ml-1m
LOG_LEVEL=INFO
```

---

## Cloud Deployment

### Deploy to Render

1. Connect GitHub repository
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`
4. Deploy!


---

## Documentation

- **API Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **MLflow UI**: http://localhost:5000
- **Dataset Info**: [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/)

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


---

## Further Reading

- MovieLens dataset by [GroupLens Research](https://grouplens.org/)
- [Surprise library](http://surpriselib.com/) for collaborative filtering
- [FastAPI](https://fastapi.tiangolo.com/) framework
- [MLflow](https://mlflow.org/) for experiment tracking

---



If this project helped you, please consider giving it a star!
