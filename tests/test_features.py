"""Feature encoders: sklearn pipeline and torch encoder."""

from __future__ import annotations

import numpy as np

from churn.features import TorchFeatureEncoder, build_sklearn_preprocessor


def test_sklearn_preprocessor_no_nan_and_shapes(tiny_df):
    X, _ = tiny_df.drop(columns=["churned"]), tiny_df["churned"]
    pre = build_sklearn_preprocessor().fit(X)
    Z = pre.transform(X)
    Z = Z.toarray() if hasattr(Z, "toarray") else np.asarray(Z)
    assert Z.shape[0] == len(X)
    assert not np.isnan(Z).any()
    names = pre.get_feature_names_out()
    assert any("contract_type" in n for n in names), "one-hot columns expected"


def test_torch_encoder_shapes_and_scaling(tiny_df):
    X = tiny_df.drop(columns=["churned"])
    enc = TorchFeatureEncoder().fit(X)
    cat, cont = enc.transform(X)

    assert cat.dtype == np.int64
    assert cont.dtype == np.float32
    assert cat.shape == (len(X), len(enc.categorical_columns))
    assert cont.shape == (len(X), len(enc.numeric_columns))
    # continuous features are standardised
    assert abs(float(cont.mean())) < 0.2
    assert abs(float(cont.std()) - 1.0) < 0.2


def test_torch_encoder_handles_unseen_categories(tiny_df):
    X = tiny_df.drop(columns=["churned"])
    enc = TorchFeatureEncoder().fit(X)

    unseen = X.iloc[[0]].copy()
    unseen["contract_type"] = "Century-long contract"
    cat, _ = enc.transform(unseen)

    for j, size in enumerate(enc.vocab_sizes):
        assert 0 <= cat[0, j] < size
    # the unseen value maps to the reserved bucket (== vocab size - 1)
    contract_idx = enc.categorical_columns.index("contract_type")
    assert cat[0, contract_idx] == enc.vocab_sizes[contract_idx] - 1


def test_torch_encoder_missing_numeric_falls_back_to_mean(tiny_df):
    X = tiny_df.drop(columns=["churned"]).copy()
    X.loc[X.index[:5], "total_charges"] = np.nan
    enc = TorchFeatureEncoder().fit(X)
    _, cont = enc.transform(X)
    assert np.isfinite(cont).all()


def test_torch_encoder_is_picklable(tiny_df):
    import pickle

    X = tiny_df.drop(columns=["churned"])
    enc = TorchFeatureEncoder().fit(X)
    reloaded = pickle.loads(pickle.dumps(enc))
    np.testing.assert_array_equal(reloaded.transform(X)[0], enc.transform(X)[0])
