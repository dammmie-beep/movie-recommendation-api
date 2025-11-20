"""
Project setup script - Creates the entire project structure.
Run this first to set up your movie recommendation project.
"""

import os
from pathlib import Path


def create_directory_structure():
    """Create all project directories."""
    directories = [
        "data/raw",
        "data/processed",
        "data/models",
        "src/data",
        "src/models",
        "src/api",
        "src/utils",
        "scripts",
        "tests",
        "notebooks",
        "docker",
        ".github/workflows",
    ]
    
    print("Creating project structure...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ Created: {directory}/")
    
    # Create __init__.py files for Python packages
    init_files = [
        "src/__init__.py",
        "src/data/__init__.py",
        "src/models/__init__.py",
        "src/api/__init__.py",
        "src/utils/__init__.py",
    ]
    
    for init_file in init_files:
        Path(init_file).touch()
        print(f"  ✅ Created: {init_file}")


def create_gitignore():
    """Create .gitignore file."""
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints
*.ipynb_checkpoints

# Environment
.env
.venv
env/
venv/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# Project specific
data/raw/
data/processed/
data/models/*.pkl
mlruns/
*.log

# OS
.DS_Store
Thumbs.db

# Model files (large)
*.pkl
*.h5
*.pth
*.onnx

# Data files (large)
*.csv
*.zip
*.gz
"""
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content)
    print("  ✅ Created: .gitignore")


def create_env_example():
    """Create .env.example file."""
    env_content = """# Model Configuration
MODEL_PATH=data/models/best_model.pkl
DATASET_PATH=data/raw/ml-1m

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Logging
LOG_LEVEL=INFO

# MLflow
MLFLOW_TRACKING_URI=file:./mlruns

# Redis (optional)
REDIS_URL=redis://localhost:6379

# Deployment
ENVIRONMENT=development
"""
    
    with open(".env.example", "w") as f:
        f.write(env_content)
    print("  ✅ Created: .env.example")


def create_readme_stub():
    """Create a basic README."""
    readme_content = """# Movie Recommendation API

    **Project in progress** 

A production-ready movie recommendation system using collaborative filtering.

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt

# Download data
python scripts/download_data.py --size 1m

# Train model
python src/train.py --quick

# Start API
uvicorn src.api.main:app --reload
```

## Progress

- [ ] Setup project structure
- [ ] Download data
- [ ] Build data loader
- [ ] Train models
- [ ] Create API
- [ ] Deploy

## Tech Stack

- Python 3.9+
- FastAPI
- Surprise (Collaborative Filtering)
- MLflow
- Docker

---

*Last updated: [Add date]*
"""
    
    with open("README.md", "w") as f:
        f.write(readme_content)
    print("  ✅ Created: README.md")


def create_dockerfile():
    """Create Dockerfile."""
    dockerfile_content = """FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ ./src/
COPY data/models/ ./data/models/
COPY data/raw/ml-1m/ ./data/raw/ml-1m/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
"""
    
    with open("docker/Dockerfile", "w") as f:
        f.write(dockerfile_content)
    print("  ✅ Created: docker/Dockerfile")


def create_docker_compose():
    """Create docker-compose.yml."""
    compose_content = """version: '3.8'

services:
  api:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=data/models/best_model.pkl
      - LOG_LEVEL=INFO
    volumes:
      - ../data:/app/data
    restart: unless-stopped
    
  # Optional: Redis for caching
  # redis:
  #   image: redis:7-alpine
  #   ports:
  #     - "6379:6379"
  #   restart: unless-stopped
"""
    
    with open("docker/docker-compose.yml", "w") as f:
        f.write(compose_content)
    print("  ✅ Created: docker/docker-compose.yml")


def create_config_file():
    """Create config.py."""
    config_content = """\"\"\"
Configuration settings for the application.
\"\"\"

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    \"\"\"Application settings.\"\"\"
    
    # Paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_PATH: Path = PROJECT_ROOT / "data"
    MODEL_PATH: Path = DATA_PATH / "models" / "best_model.pkl"
    
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_TITLE: str = "Movie Recommendation API"
    API_VERSION: str = "1.0.0"
    
    # Model
    DEFAULT_N_RECOMMENDATIONS: int = 10
    MAX_N_RECOMMENDATIONS: int = 100
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    # MLflow
    MLFLOW_TRACKING_URI: str = "file:./mlruns"
    MLFLOW_EXPERIMENT_NAME: str = "movie-recommendation"
    
    class Config:
        env_file = ".env"


settings = Settings()
"""
    
    with open("src/config.py", "w") as f:
        f.write(config_content)
    print("  ✅ Created: src/config.py")


def create_pytest_config():
    """Create pytest configuration."""
    pytest_content = """[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --strict-markers
    --tb=short
    --cov=src
    --cov-report=html
    --cov-report=term-missing
"""
    
    with open("pytest.ini", "w") as f:
        f.write(pytest_content)
    print("  ✅ Created: pytest.ini")


def create_github_actions():
    """Create GitHub Actions CI workflow."""
    workflow_content = """name: CI Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Lint with flake8
      run: |
        pip install flake8
        flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics
    
    - name: Run tests
      run: |
        pytest tests/ -v --cov=src
    
    - name: Check code formatting
      run: |
        pip install black
        black --check src/

  build-docker:
    runs-on: ubuntu-latest
    needs: test
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Build Docker image
      run: |
        docker build -t movie-rec-api:latest -f docker/Dockerfile .
"""
    
    with open(".github/workflows/ci.yml", "w") as f:
        f.write(workflow_content)
    print("  ✅ Created: .github/workflows/ci.yml")


def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "="*60)
    print("🎉 Project structure created successfully!")
    print("="*60)
    
    print("\n📋 Next Steps:")
    print("\n1️⃣  Copy the artifact files into your project:")
    print("   - requirements.txt")
    print("   - scripts/download_data.py")
    print("   - src/data/loader.py")
    print("   - src/models/collaborative.py")
    print("   - src/api/main.py")
    print("   - src/train.py")
    
    print("\n2️⃣  Set up Python environment:")
    print("   python -m venv venv")
    print("   source venv/bin/activate  # Windows: venv\\Scripts\\activate")
    print("   pip install -r requirements.txt")
    
    print("\n3️⃣  Download data:")
    print("   python scripts/download_data.py --size 1m")
    
    print("\n4️⃣  Test data loading:")
    print("   python src/data/loader.py")
    
    print("\n5️⃣  Train your first model:")
    print("   python src/train.py --quick --sample 50000")
    
    print("\n6️⃣  Start the API:")
    print("   uvicorn src.api.main:app --reload")
    
    print("\n📚 Reference the Week-by-Week Checklist for detailed guidance!")
    print("="*60)


def main():
    """Main setup function."""
    print("\n" + "="*60)
    print("Movie Recommendation API - Project Setup")
    print("="*60)
    print()
    
    # Check if project already exists
    if Path("src").exists():
        response = input("⚠️  Project structure already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Setup cancelled.")
            return
    
    try:
        create_directory_structure()
        create_gitignore()
        create_env_example()
        create_readme_stub()
        create_dockerfile()
        create_docker_compose()
        create_config_file()
        create_pytest_config()
        create_github_actions()
        
        print_next_steps()
        
    except Exception as e:
        print(f"\n❌ Error during setup: {e}")
        raise


if __name__ == "__main__":
    main()