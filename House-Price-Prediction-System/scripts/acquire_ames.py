"""Download the public Ames Housing source into datasets/raw (network required)."""
from pathlib import Path
from sklearn.datasets import fetch_openml
import pandas as pd
from house_price.data.validation import FEATURES, TARGET

def main():
    data = fetch_openml(name="house_prices", as_frame=True, parser="auto").frame
    subset = data[FEATURES + [TARGET]].copy(); subset.to_csv(Path(__file__).parents[1] / "datasets/raw/ames_subset.csv", index=False)
if __name__ == "__main__": main()
