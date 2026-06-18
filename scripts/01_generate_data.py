"""Step 1: generate the synthetic rainfall-runoff dataset.

    python scripts/01_generate_data.py            # 20 years -> data/synthetic/basin.csv
"""
import argparse
from pathlib import Path

from lstm_hydrology import make_dataset


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--years", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("data/synthetic/basin.csv"))
    args = ap.parse_args()

    df = make_dataset(n_years=args.years, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out)
    print(f"Wrote {len(df):,} daily records to {args.out}\n")
    print(df.describe().round(2))


if __name__ == "__main__":
    main()
