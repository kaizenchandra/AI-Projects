from pathlib import Path
import numpy as np, pandas as pd

def main():
    rng = np.random.default_rng(42); n = 240
    quality = rng.integers(3, 11, n); area = rng.normal(1600, 500, n).clip(400, 4500)
    neighborhoods = rng.choice(["CollgCr", "OldTown", "NridgHt", "Edwards"], n)
    effects = {"CollgCr": 25000, "OldTown": -15000, "NridgHt": 50000, "Edwards": -5000}
    price = 30000 + quality*18000 + area*85 + rng.normal(0, 22000, n) + np.array([effects[x] for x in neighborhoods])
    frame = pd.DataFrame({"OverallQual": quality, "GrLivArea": area.round(1), "GarageCars": rng.integers(0, 4, n), "TotalBsmtSF": rng.normal(1050, 350, n).clip(0, 2500).round(1), "YearBuilt": rng.integers(1950, 2021, n), "Neighborhood": neighborhoods, "SalePrice": price.clip(25000).round(2)})
    target = Path(__file__).parents[1] / "datasets/raw/ames_subset.csv"; target.parent.mkdir(parents=True, exist_ok=True); frame.to_csv(target, index=False); print(target)
if __name__ == "__main__": main()
