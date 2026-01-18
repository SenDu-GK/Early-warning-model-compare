import os
import numpy as np
import pandas as pd

def main():
    os.makedirs("data/raw", exist_ok=True)
    rng = np.random.default_rng(42)

    n = 800
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)

    # 简单二分类：线性可分 + 噪声
    logits = 1.2 * x1 - 0.8 * x2 + 0.5 * x3 + rng.normal(scale=0.4, size=n)
    y = (logits > 0).astype(int)

    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "y": y})
    df.to_csv("data/raw/toy.csv", index=False)
    print("Saved: data/raw/toy.csv")

if __name__ == "__main__":
    main()
