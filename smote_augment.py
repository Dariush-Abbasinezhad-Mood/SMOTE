#!/usr/bin/env python3
"""
smote_augment.py

Reads a CSV, imputes missing values, encodes categoricals, scales numerics,
applies SMOTE or SMOTENC to oversample the minority class by a given factor,
then inverts transforms and writes out an augmented CSV.
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from imblearn.over_sampling import SMOTE, SMOTENC

def parse_args():
    parser = argparse.ArgumentParser(
        description="Augment minority class via SMOTE/SMOTENC"
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to the input CSV file"
    )
    parser.add_argument(
        "--factor", "-f", type=float, required=True,
        help="Oversampling factor for the minority class (e.g., 2.0 doubles)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    df = pd.read_csv(args.input)

    # Split features and target (assume last column is the target)
    feature_cols = df.columns[:-1].tolist()
    target_col = df.columns[-1]
    X_raw = df[feature_cols].copy()
    y = df[target_col].copy()

    # Identify numeric and categorical columns
    cat_cols = X_raw.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in feature_cols if c not in cat_cols]

    # Impute missing values
    if num_cols:
        num_imputer = SimpleImputer(strategy="mean")
        X_raw[num_cols] = num_imputer.fit_transform(X_raw[num_cols])

    if cat_cols:
        cat_imputer = SimpleImputer(strategy="constant", fill_value="__MISSING__")
        X_raw[cat_cols] = cat_imputer.fit_transform(X_raw[cat_cols])

    # Encode categorical features
    if cat_cols:
        enc = OrdinalEncoder()
        X_cat = enc.fit_transform(X_raw[cat_cols])
    else:
        X_cat = np.empty((len(X_raw), 0))

    # Scale numeric features
    if num_cols:
        scaler = StandardScaler()
        X_num = scaler.fit_transform(X_raw[num_cols])
    else:
        X_num = np.empty((len(X_raw), 0))

    # Combine numeric and categorical arrays
    X_enc = np.hstack([X_num, X_cat])

    # Determine minority class and sampling strategy
    counts = y.value_counts()
    minority_class = counts.idxmin()
    orig_count = counts.min()
    new_count = int(round(orig_count * args.factor))
    if new_count <= orig_count:
        raise ValueError(f"Factor must be > 1.0; got {args.factor}")
    sampling_strategy = {minority_class: new_count}

    # Choose SMOTE variant
    if cat_cols:
        cat_indices = list(range(len(num_cols), len(num_cols) + len(cat_cols)))
        sampler = SMOTENC(
            categorical_features=cat_indices,
            sampling_strategy=sampling_strategy,
            random_state=42
        )
    else:
        sampler = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=42
        )

    # Apply resampling
    X_res, y_res = sampler.fit_resample(X_enc, y.values)

    # Inverse transform numeric features
    if num_cols:
        X_num_res = X_res[:, :len(num_cols)]
        df_num = pd.DataFrame(
            scaler.inverse_transform(X_num_res),
            columns=num_cols
        )
    else:
        df_num = pd.DataFrame()

    # Inverse transform categorical features
    if cat_cols:
        X_cat_res = X_res[:, len(num_cols):]
        df_cat = pd.DataFrame(
            enc.inverse_transform(X_cat_res),
            columns=cat_cols
        )
    else:
        df_cat = pd.DataFrame()

    # Combine back into one DataFrame
    df_out = pd.concat([df_num, df_cat], axis=1)
    df_out[target_col] = y_res

    # Save augmented dataset
    base, ext = os.path.splitext(args.input)
    out_path = f"{base}_augmented{ext}"
    df_out.to_csv(out_path, index=False)

    print(f"Original minority '{minority_class}': {orig_count} samples")
    print(f"Augmented minority '{minority_class}': {new_count} samples")
    print(f"Augmented CSV written to: {out_path}")

if __name__ == "__main__":
    main()
