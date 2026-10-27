import json
from pathlib import Path
import matplotlib.pyplot as plt, seaborn as sns
from house_price.training.train import load_data

def main():
    frame = load_data(); out = Path("reports"); out.mkdir(exist_ok=True)
    out.joinpath("data_quality.json").write_text(json.dumps({"shape": frame.shape, "missing": frame.isna().sum().to_dict(), "describe": frame.describe(include="all").fillna("").to_dict()}, indent=2, default=str))
    sns.histplot(frame["SalePrice"]); plt.title("Sale price distribution"); plt.tight_layout(); plt.savefig(out/"price_distribution.png"); plt.close()
    sns.heatmap(frame.select_dtypes("number").corr(), annot=True, cmap="vlag"); plt.tight_layout(); plt.savefig(out/"correlation_heatmap.png"); plt.close()
if __name__ == "__main__": main()
