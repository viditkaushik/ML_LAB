import os
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def make_breast_cancer_csv(out_path: str) -> None:
    data = load_breast_cancer(as_frame=True)
    df = data.frame
    df.to_csv(out_path, index=False)


def make_iris_csv(out_path: str) -> None:
    iris = load_iris(as_frame=True)
    df = iris.frame
    df.to_csv(out_path, index=False)


def main() -> None:
    base = os.path.dirname(__file__)
    ensure_dir(base)

    make_breast_cancer_csv(os.path.join(base, "breast_cancer.csv"))
    make_iris_csv(os.path.join(base, "iris.csv"))

    print("CSV files created under:", base)


if __name__ == "__main__":
    main()

