"""
Generate synthetic_churn.csv — 1000 rows, binary churn target.

Hidden signal: churn is strongly predicted by
    (income / account_balance) * (1 / (last_contact_days + 1))
plus Gaussian noise. The agent should discover a ratio feature and a
recency-decay feature to recover this signal.
"""
import pathlib

import numpy as np
import pandas as pd


def generate(seed: int = 42, n: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 75, n)
    income = rng.integers(20_000, 120_000, n).astype(float)
    account_balance = rng.integers(500, 50_000, n).astype(float)
    city_code = rng.integers(1, 20, n)
    num_products = rng.integers(1, 6, n)
    years_as_customer = rng.integers(0, 30, n)
    last_contact_days = rng.integers(1, 365, n)
    num_contacts = rng.integers(1, 20, n)
    prev_outcome = rng.integers(0, 3, n)        # 0=none, 1=failure, 2=success
    marital_status_code = rng.integers(0, 3, n) # 0=single, 1=married, 2=divorced
    education_code = rng.integers(0, 4, n)
    job_code = rng.integers(0, 12, n)

    # Hidden signal: interaction of ratio and recency
    signal = (income / account_balance) * (1.0 / (last_contact_days + 1))
    noise = rng.standard_normal(n) * signal.std() * 0.5

    logit = 2.0 * (signal - signal.mean()) / signal.std() + noise
    prob_churn = 1.0 / (1.0 + np.exp(-logit))
    churn = (rng.random(n) < prob_churn).astype(int)

    return pd.DataFrame({
        "age": age,
        "income": income,
        "account_balance": account_balance,
        "city_code": city_code,
        "num_products": num_products,
        "years_as_customer": years_as_customer,
        "last_contact_days": last_contact_days,
        "num_contacts": num_contacts,
        "prev_outcome": prev_outcome,
        "marital_status_code": marital_status_code,
        "education_code": education_code,
        "job_code": job_code,
        "churn": churn,
    })


if __name__ == "__main__":
    df = generate()
    out = pathlib.Path(__file__).parent / "synthetic_churn.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}")
