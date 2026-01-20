import pytest
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import (
    train_model,
    compute_model_metrics,
    inference,
    save_model,
    load_model,
    performance_on_categorical_slice,
)

# Keep consistent with train_model.py
CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

@pytest.fixture(scope="module")
def census_data():
    """
    Loads census.csv and returns (train_df, test_df).
    Uses the same relative path logic as train_model.py.

    Parameters: None
    Return: pd.DataFrame, pd.DataFrame → train_df, test_df
    """
    project_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(project_path, "data", "census.csv")
    df = pd.read_csv(data_path, skipinitialspace=True)

    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df["salary"] if "salary" in df.columns else None,
    )
    return train_df, test_df


@pytest.fixture(scope="module")
def trained_artifacts(census_data):
    """
    Trains a model once for the module and returns
    (model, encoder, lb, X_test, y_test, test_df).

    Parameters: census_data
    Return: model, encoder, lb, X_test, y_test, test_df
    """
    train_df, test_df = census_data

    X_train, y_train, encoder, lb = process_data(
        train_df,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=True,
    )

    X_test, y_test, _, _ = process_data(
        test_df,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    model = train_model(X_train, y_train)
    return model, encoder, lb, X_test, y_test, test_df


def test_process_data_shapes_and_outputs(census_data):
    """
    Test that process_data returns properly shaped arrays and fitted encoder/lb in training mode.
    
    Parameters: census_data
    """
    train_df, _ = census_data

    X_train, y_train, encoder, lb = process_data(
        train_df,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=True,
    )

    # Basic sanity checks
    assert isinstance(X_train, np.ndarray)
    assert isinstance(y_train, np.ndarray)

    # Same number of rows
    assert X_train.shape[0] == y_train.shape[0]
    assert X_train.shape[0] == train_df.shape[0]

    # Encoder and label binarizer should be created in training mode
    assert encoder is not None
    assert lb is not None

    # Binary labels expected
    assert set(np.unique(y_train)).issubset({0, 1})


def test_train_and_inference_and_metrics(trained_artifacts):
    """
    Test that the trained model can run inference and produce valid metrics.
    
    Parameters: trained_artifacts
    """
    model, encoder, lb, X_test, y_test, _ = trained_artifacts

    preds = inference(model, X_test)

    # Predictions should align in length
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == y_test.shape[0]

    # Predictions should be binary labels
    assert set(np.unique(preds)).issubset({0, 1})

    p, r, fb = compute_model_metrics(y_test, preds)

    # Metrics should be valid probabilities
    assert 0.0 <= p <= 1.0
    assert 0.0 <= r <= 1.0
    assert 0.0 <= fb <= 1.0


def test_slice_metrics_and_model_persistence(trained_artifacts, tmp_path):
    """
    Test that slice metrics are computed and that saving/loading preserves inference behavior.
    
    Parameters: trained_artifacts, tmp_path
    """
    model, encoder, lb, X_test, y_test, test_df = trained_artifacts

    # Pick a slice that should exist
    col = "sex"
    slice_value = test_df[col].dropna().unique()[0]

    p_s, r_s, fb_s = performance_on_categorical_slice(
        data=test_df,
        column_name=col,
        slice_value=slice_value,
        categorical_features=CAT_FEATURES,
        label="salary",
        encoder=encoder,
        lb=lb,
        model=model,
    )

    assert 0.0 <= p_s <= 1.0
    assert 0.0 <= r_s <= 1.0
    assert 0.0 <= fb_s <= 1.0

    # Test save/load round-trip
    model_path = tmp_path / "model.pkl"
    save_model(model, str(model_path))
    loaded_model = load_model(str(model_path))

    preds_before = inference(model, X_test)
    preds_after = inference(loaded_model, X_test)

    # Same predictions after reload
    assert np.array_equal(preds_before, preds_after)