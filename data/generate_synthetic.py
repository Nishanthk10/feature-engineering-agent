"""
Generate synthetic datasets for benchmarking.

synthetic_churn.csv — 1000 rows, binary churn target.
Hidden signal: churn is strongly predicted by
    (income / account_balance) * (1 / (last_contact_days + 1))
plus Gaussian noise. The agent should discover a ratio feature and a
recency-decay feature to recover this signal.

synthetic_regression.csv — 1000 rows, continuous price target.
Hidden signal: price is strongly driven by
    (sqft / age_years) * condition_score + (1 / distance_to_center) * 10000
plus Gaussian noise. The agent should discover an area-efficiency ratio and
a location-decay feature to recover this signal.
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


def generate_regression_dataset(seed: int = 42, n: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    sqft = rng.integers(500, 4000, n).astype(float)
    bedrooms = rng.integers(1, 6, n)
    bathrooms = rng.integers(1, 4, n)
    age_years = rng.integers(1, 80, n).astype(float)
    distance_to_center = rng.uniform(0.5, 30.0, n)
    neighbourhood_code = rng.integers(1, 15, n)
    condition_score = rng.uniform(1.0, 10.0, n)
    garage = rng.integers(0, 2, n)
    floors = rng.integers(1, 4, n)

    # Hidden signal: area-efficiency ratio + location decay
    area_efficiency = (sqft / age_years) * condition_score
    location_decay = (1.0 / distance_to_center) * 10_000

    base_price = 50_000 + area_efficiency * 20.0 + location_decay
    noise = rng.standard_normal(n) * base_price.std() * 0.15
    price = (base_price + noise).clip(min=20_000)

    return pd.DataFrame({
        "sqft": sqft,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "age_years": age_years,
        "distance_to_center": distance_to_center,
        "neighbourhood_code": neighbourhood_code,
        "condition_score": condition_score,
        "garage": garage,
        "floors": floors,
        "price": price,
    })


if __name__ == "__main__":
    data_dir = pathlib.Path(__file__).parent

    df_churn = generate()
    out_churn = data_dir / "synthetic_churn.csv"
    df_churn.to_csv(out_churn, index=False)
    print(f"Saved {len(df_churn)} rows to {out_churn}")

    df_reg = generate_regression_dataset()
    out_reg = data_dir / "synthetic_regression.csv"
    df_reg.to_csv(out_reg, index=False)
    print(f"Saved {len(df_reg)} rows to {out_reg}")
