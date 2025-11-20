"""
Download MovieLens dataset for the recommendation system.
Supports both 1M (smaller, for development) and 25M (production) datasets.
"""

import os
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm


def download_file(url: str, destination: Path):
    """Download a file with progress bar."""
    print(f"Downloading from {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as file, tqdm(
        desc=destination.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            pbar.update(size)
    
    print(f"✅ Download complete: {destination}")


def extract_zip(zip_path: Path, extract_to: Path):
    """Extract zip file."""
    print(f"\nExtracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("✅ Extraction complete!")


def download_movielens(size: str = "1m"):
    """
    Download MovieLens dataset.
    
    Args:
        size: Either "1m" (1 million ratings, ~6MB) or "25m" (25 million ratings, ~250MB)
    """
    # Create data directory
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Dataset URLs
    datasets = {
        "1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
        "25m": "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
    }
    
    if size not in datasets:
        raise ValueError(f"Size must be '1m' or '25m', got '{size}'")
    
    url = datasets[size]
    zip_filename = f"ml-{size}.zip"
    zip_path = data_dir / zip_filename
    
    # Check if already extracted
    dataset_dir = data_dir / f"ml-{size}"
    if dataset_dir.exists():
        print(f"✅ Dataset already exists at: {dataset_dir}")
        print("\nDataset files:")
        for item in sorted(dataset_dir.iterdir()):
            if item.is_file():
                size_mb = item.stat().st_size / (1024 * 1024)
                print(f"  - {item.name} ({size_mb:.2f} MB)")
        return
    
    # Download if not already present
    if not zip_path.exists():
        print(f"\n📥 Downloading MovieLens {size.upper()} dataset...")
        try:
            download_file(url, zip_path)
        except Exception as e:
            print(f"❌ Download failed: {e}")
            if zip_path.exists():
                zip_path.unlink()
            raise
    else:
        print(f"✅ Dataset already downloaded: {zip_path}")
    
    # Extract
    try:
        extract_zip(zip_path, data_dir)
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        raise
    
    # Verify extraction
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Expected directory not found after extraction: {dataset_dir}")
    
    # Show dataset info
    print(f"\n✅ Dataset ready at: {dataset_dir}")
    print("\nDataset structure:")
    for item in sorted(dataset_dir.iterdir()):
        if item.is_file():
            size_mb = item.stat().st_size / (1024 * 1024)
            print(f"  - {item.name} ({size_mb:.2f} MB)")
    
    # Optional: Clean up zip file
    print(f"\nZip file kept at: {zip_path}")
    print("(You can delete it manually to save space)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download MovieLens dataset")
    parser.add_argument(
        "--size",
        type=str,
        default="1m",
        choices=["1m", "25m"],
        help="Dataset size: 1m (development) or 25m (production)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MovieLens Dataset Downloader")
    print("=" * 60)
    print(f"\nDataset: MovieLens {args.size.upper()}")
    
    if args.size == "25m":
        print("\n⚠️  Warning: 25M dataset is ~250MB and takes longer to process.")
        print("For development, consider using --size 1m first.")
        response = input("\nContinue with 25M? (y/n): ")
        if response.lower() != 'y':
            print("\n✅ Switching to 1M dataset...")
            args.size = "1m"
    
    try:
        download_movielens(args.size)
        
        print("\n" + "=" * 60)
        print("🎉 Success! Next steps:")
        print("=" * 60)
        print("1. Explore data: python src/data/loader.py")
        print("2. Train models: python src/train.py --quick")
        print("3. Start API: uvicorn src.api.main:app --reload")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("- Check your internet connection")
        print("- Ensure you have write permissions in the current directory")
        print("- Try running with sudo/admin rights if needed")
        exit(1)