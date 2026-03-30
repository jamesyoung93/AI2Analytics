"""Stage E: Model training & backtesting — probability, depth, and look-alike models."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_score,
    recall_score, f1_score, accuracy_score, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score,
)

from ai2analytics.templates.detail_optimization.config import DetailOptimizationConfig
from ai2analytics.templates.detail_optimization.features import FeatureSet
from ai2analytics.utils import safe_fill


@dataclass
class TrainedModels:
    """Container for trained models and their backtest metrics."""
    prob_model: RandomForestClassifier | None = None
    depth_model: RandomForestRegressor | None = None
    look_model: RandomForestClassifier | None = None
    metrics_prob: pd.DataFrame = field(default_factory=pd.DataFrame)
    metrics_depth: pd.DataFrame = field(default_factory=pd.DataFrame)
    metrics_look: pd.DataFrame = field(default_factory=pd.DataFrame)


def train_models(
    cfg: DetailOptimizationConfig,
    features: FeatureSet,
) -> TrainedModels:
    """Train all three models with backtesting, then retrain on full data."""
    print("=" * 70)
    print("STAGE E: Model training & backtesting")
    print("=" * 70)

    result = TrainedModels()
    col_week = cfg.col_week
    col_npi = cfg.col_npi

    # E1. General probability model
    print("\n  -- E1: General probability model --")
    result.prob_model, result.metrics_prob = _train_classifier(
        df=features.df,
        feat_cols=features.feat_prob,
        target_col="target_prob",
        col_week=col_week,
        col_npi=col_npi,
        model_params=cfg.prob_model_params,
        cfg=cfg,
        label="Prob",
    )

    # E2. Depth model (regression, positives only)
    print("\n  -- E2: Depth model (regression) --")
    df_depth = features.df[features.df["target_cnt"] > 0].copy()
    df_depth = safe_fill(df_depth, features.feat_depth)
    result.depth_model, result.metrics_depth = _train_regressor(
        df=df_depth,
        feat_cols=features.feat_depth,
        target_col="target_cnt",
        col_week=col_week,
        model_params=cfg.depth_model_params,
        cfg=cfg,
        label="Depth",
    )

    # E3. Look-alike model (never-writers only)
    print("\n  -- E3: Look-alike model --")
    look_mask = (
        features.df["first_write_week"].isna()
        | (features.df["first_write_week"] > features.df[col_week])
    )
    df_look = features.df[look_mask].copy()
    df_look = safe_fill(df_look, features.feat_look)
    result.look_model, result.metrics_look = _train_classifier(
        df=df_look,
        feat_cols=features.feat_look,
        target_col="target_look",
        col_week=col_week,
        col_npi=col_npi,
        model_params=cfg.look_model_params,
        cfg=cfg,
        label="Look",
        look_alike=True,
    )

    print("  Done.\n")
    return result


def _get_backtest_folds(
    df: pd.DataFrame, col_week: str, cfg: DetailOptimizationConfig
) -> list:
    unique_wks = np.sort(df[col_week].unique())
    max_week = df[col_week].max()
    cutoff = max_week - pd.Timedelta(weeks=cfg.backtest_gap_weeks)
    return [w for w in unique_wks if w <= cutoff][-cfg.n_backtest_folds:]


def _train_classifier(
    df: pd.DataFrame,
    feat_cols: list[str],
    target_col: str,
    col_week: str,
    col_npi: str,
    model_params: dict,
    cfg: DetailOptimizationConfig,
    label: str,
    look_alike: bool = False,
) -> tuple[RandomForestClassifier | None, pd.DataFrame]:
    """Backtest a classifier, then retrain on full data."""
    horizon = cfg.target_horizon_weeks
    train_ends = _get_backtest_folds(df, col_week, cfg)
    unique_wks = np.sort(df[col_week].unique())

    metrics = []
    for idx, train_end in enumerate(train_ends, start=1):
        test_week = train_end + pd.Timedelta(weeks=horizon)
        if test_week not in unique_wks:
            continue

        if look_alike:
            train_mask = (
                (df[col_week] <= train_end)
                & (df["first_write_week"].isna() | (df[col_week] <= df["first_write_week"]))
            )
            test_mask = (
                (df[col_week] == test_week)
                & (df["first_write_week"].isna() | (df["first_write_week"] > test_week))
            )
        else:
            train_mask = df[col_week] <= train_end
            test_mask = df[col_week] == test_week

        X_train = df.loc[train_mask, feat_cols]
        y_train = df.loc[train_mask, target_col]
        X_test = df.loc[test_mask, feat_cols]
        y_test = df.loc[test_mask, target_col]

        if y_test.nunique() < 2:
            print(f"    -> fold {idx}: skipped (single class in test)")
            continue

        clf = RandomForestClassifier(**model_params)
        clf.fit(X_train, y_train)

        proba = clf.predict_proba(X_test)[:, 1]
        pred = (proba >= cfg.prob_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

        metrics.append({
            "fold": idx,
            "train_end": pd.Timestamp(train_end).strftime("%Y-%m-%d"),
            "test_week": pd.Timestamp(test_week).strftime("%Y-%m-%d"),
            "pr_auc": average_precision_score(y_test, proba),
            "roc_auc": roc_auc_score(y_test, proba),
            "precision": precision_score(y_test, pred),
            "recall": recall_score(y_test, pred),
            "f1": f1_score(y_test, pred),
            "accuracy": accuracy_score(y_test, pred),
            "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0,
            "random_chance": y_test.mean(),
            "test_size": len(y_test),
        })
        print(f"    fold {idx}  PR-AUC={metrics[-1]['pr_auc']:.4f}")

    metrics_df = pd.DataFrame(metrics)
    if not metrics_df.empty:
        print(f"  {label} avg PR-AUC: {metrics_df['pr_auc'].mean():.4f}")

    # Retrain on full data for production scoring
    final_model = None
    if cfg.retrain_on_full_data and len(df) > 0:
        final_model = RandomForestClassifier(**model_params)
        final_model.fit(df[feat_cols], df[target_col])
        print(f"  {label} model retrained on full data ({len(df):,} rows)")

    return final_model, metrics_df


def _train_regressor(
    df: pd.DataFrame,
    feat_cols: list[str],
    target_col: str,
    col_week: str,
    model_params: dict,
    cfg: DetailOptimizationConfig,
    label: str,
) -> tuple[RandomForestRegressor | None, pd.DataFrame]:
    """Backtest a regressor, then retrain on full data."""
    horizon = cfg.target_horizon_weeks
    train_ends = _get_backtest_folds(df, col_week, cfg)
    unique_wks = np.sort(df[col_week].unique())

    metrics = []
    for idx, train_end in enumerate(train_ends, start=1):
        test_week = train_end + pd.Timedelta(weeks=horizon)
        if test_week not in unique_wks:
            continue

        train_mask = df[col_week] <= train_end
        test_mask = df[col_week] == test_week

        X_train = df.loc[train_mask, feat_cols]
        y_train = df.loc[train_mask, target_col]
        X_test = df.loc[test_mask, feat_cols]
        y_test = df.loc[test_mask, target_col]

        if y_test.empty:
            continue

        reg = RandomForestRegressor(**model_params)
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)

        metrics.append({
            "fold": idx,
            "train_end": pd.Timestamp(train_end).strftime("%Y-%m-%d"),
            "test_week": pd.Timestamp(test_week).strftime("%Y-%m-%d"),
            "MAE": mean_absolute_error(y_test, y_pred),
            "MSE": mean_squared_error(y_test, y_pred),
            "R2": r2_score(y_test, y_pred),
        })
        print(f"    fold {idx}  MAE={metrics[-1]['MAE']:.4f}  R2={metrics[-1]['R2']:.4f}")

    metrics_df = pd.DataFrame(metrics)
    if not metrics_df.empty:
        print(f"  {label} avg MAE: {metrics_df['MAE'].mean():.4f}")

    # Retrain on full data
    final_model = None
    if cfg.retrain_on_full_data and len(df) > 0:
        final_model = RandomForestRegressor(**model_params)
        final_model.fit(df[feat_cols], df[target_col])
        print(f"  {label} model retrained on full data ({len(df):,} rows)")

    return final_model, metrics_df


def print_feature_importance(models: TrainedModels, features: FeatureSet, top_n: int = 20):
    """Print top feature importances for each model."""
    print("\n" + "=" * 70)
    print(f"TOP {top_n} FEATURE IMPORTANCES")
    print("=" * 70)

    for name, model, feat_cols in [
        ("Probability", models.prob_model, features.feat_prob),
        ("Depth", models.depth_model, features.feat_depth),
        ("Look-alike", models.look_model, features.feat_look),
    ]:
        if model is not None:
            imp = (
                pd.Series(model.feature_importances_, index=feat_cols)
                .sort_values(ascending=False)
                .head(top_n)
            )
            print(f"\n  {name} Model:")
            for feat, val in imp.items():
                print(f"    {feat:45s} {val:.4f}")
