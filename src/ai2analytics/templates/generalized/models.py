"""Cadence-aware model training for generalized detail optimization."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from ai2analytics.templates.detail_optimization.features import FeatureSet
from ai2analytics.templates.generalized.cadence import add_periods, resolve_existing_period
from ai2analytics.templates.generalized.config import GeneralizedDetailOptimizationConfig
from ai2analytics.templates.generalized.reporting import GeneralizedReporter
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
    cfg: GeneralizedDetailOptimizationConfig,
    features: FeatureSet,
    reporter: GeneralizedReporter | None = None,
) -> TrainedModels:
    """Train all three models with cadence-aware backtesting."""
    reporter = reporter or GeneralizedReporter(cfg.verbose, cfg.warn_on_defaults)

    print("=" * 70)
    print("STAGE E: Model training & backtesting (generalized)")
    print("=" * 70)

    result = TrainedModels()
    col_week = cfg.col_week
    col_npi = cfg.col_npi

    print("\n  -- E1: General probability model --")
    result.prob_model, result.metrics_prob = _train_classifier(
        df=features.df,
        feat_cols=features.feat_prob,
        target_col="target_prob",
        col_week=col_week,
        col_npi=col_npi,
        model_params=cfg.prob_model_params,
        cfg=cfg,
        reporter=reporter,
        label="Prob",
    )

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
        reporter=reporter,
        label="Depth",
    )

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
        reporter=reporter,
        label="Look",
        look_alike=True,
    )

    print("  Done.\n")
    return result


def _get_backtest_folds(
    df: pd.DataFrame,
    col_week: str,
    cfg: GeneralizedDetailOptimizationConfig,
) -> list[pd.Timestamp]:
    unique_periods = np.sort(pd.to_datetime(df[col_week].unique()))
    if len(unique_periods) == 0:
        return []
    max_period = pd.Timestamp(df[col_week].max())
    cutoff = add_periods(max_period, -cfg.effective_backtest_gap_periods(), cfg.planning_cadence)
    return [pd.Timestamp(p) for p in unique_periods if pd.Timestamp(p) <= cutoff][
        -cfg.n_backtest_folds:
    ]


def _resolve_test_period(
    unique_periods,
    target,
    reporter: GeneralizedReporter,
    label: str,
    fold: int,
) -> pd.Timestamp | None:
    resolved, fallback = resolve_existing_period(unique_periods, target, prefer="on_or_after")
    if resolved is None:
        reporter.warn(
            f"{label} fold {fold}: no observed test period on or after target {target}; "
            "skipping this fold."
        )
        return None
    if fallback:
        reporter.warn(
            f"{label} fold {fold}: exact test period {pd.Timestamp(target).date()} was "
            f"not present; using next observed period {resolved.date()}."
        )
    return resolved


def _train_classifier(
    df: pd.DataFrame,
    feat_cols: list[str],
    target_col: str,
    col_week: str,
    col_npi: str,
    model_params: dict,
    cfg: GeneralizedDetailOptimizationConfig,
    reporter: GeneralizedReporter,
    label: str,
    look_alike: bool = False,
) -> tuple[RandomForestClassifier | None, pd.DataFrame]:
    """Backtest a classifier, then retrain on full data."""
    horizon = cfg.effective_horizon_periods()
    train_ends = _get_backtest_folds(df, col_week, cfg)
    unique_periods = np.sort(pd.to_datetime(df[col_week].unique()))

    metrics = []
    for idx, train_end in enumerate(train_ends, start=1):
        target_test_period = add_periods(train_end, horizon, cfg.planning_cadence)
        test_period = _resolve_test_period(unique_periods, target_test_period, reporter, label, idx)
        if test_period is None:
            continue

        if look_alike:
            train_mask = (
                (df[col_week] <= train_end)
                & (df["first_write_week"].isna() | (df[col_week] <= df["first_write_week"]))
            )
            test_mask = (
                (df[col_week] == test_period)
                & (df["first_write_week"].isna() | (df["first_write_week"] > test_period))
            )
        else:
            train_mask = df[col_week] <= train_end
            test_mask = df[col_week] == test_period

        X_train = df.loc[train_mask, feat_cols]
        y_train = df.loc[train_mask, target_col]
        X_test = df.loc[test_mask, feat_cols]
        y_test = df.loc[test_mask, target_col]

        if y_train.empty or y_train.nunique() < 2:
            reporter.warn(f"{label} fold {idx}: skipped because training data has one class.")
            continue
        if y_test.empty or y_test.nunique() < 2:
            print(f"    -> fold {idx}: skipped (single class or empty test)")
            continue

        clf = RandomForestClassifier(**model_params)
        clf.fit(X_train, y_train)

        proba = _positive_class_probability(clf, X_test)
        pred = (proba >= cfg.prob_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

        metrics.append({
            "fold": idx,
            "train_end": pd.Timestamp(train_end).strftime("%Y-%m-%d"),
            "test_period": pd.Timestamp(test_period).strftime("%Y-%m-%d"),
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

    final_model = None
    if cfg.retrain_on_full_data and len(df) > 0:
        y_full = df[target_col]
        if y_full.nunique() < 2:
            reporter.warn(
                f"{label} model was not retrained because the full target has one class."
            )
        else:
            final_model = RandomForestClassifier(**model_params)
            final_model.fit(df[feat_cols], y_full)
            print(f"  {label} model retrained on full data ({len(df):,} rows)")

    return final_model, metrics_df


def _train_regressor(
    df: pd.DataFrame,
    feat_cols: list[str],
    target_col: str,
    col_week: str,
    model_params: dict,
    cfg: GeneralizedDetailOptimizationConfig,
    reporter: GeneralizedReporter,
    label: str,
) -> tuple[RandomForestRegressor | None, pd.DataFrame]:
    """Backtest a regressor, then retrain on full data."""
    horizon = cfg.effective_horizon_periods()
    train_ends = _get_backtest_folds(df, col_week, cfg)
    unique_periods = np.sort(pd.to_datetime(df[col_week].unique()))

    metrics = []
    for idx, train_end in enumerate(train_ends, start=1):
        target_test_period = add_periods(train_end, horizon, cfg.planning_cadence)
        test_period = _resolve_test_period(unique_periods, target_test_period, reporter, label, idx)
        if test_period is None:
            continue

        train_mask = df[col_week] <= train_end
        test_mask = df[col_week] == test_period

        X_train = df.loc[train_mask, feat_cols]
        y_train = df.loc[train_mask, target_col]
        X_test = df.loc[test_mask, feat_cols]
        y_test = df.loc[test_mask, target_col]

        if y_train.empty or y_test.empty:
            reporter.warn(f"{label} fold {idx}: skipped because train or test data is empty.")
            continue

        reg = RandomForestRegressor(**model_params)
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)

        metrics.append({
            "fold": idx,
            "train_end": pd.Timestamp(train_end).strftime("%Y-%m-%d"),
            "test_period": pd.Timestamp(test_period).strftime("%Y-%m-%d"),
            "MAE": mean_absolute_error(y_test, y_pred),
            "MSE": mean_squared_error(y_test, y_pred),
            "R2": r2_score(y_test, y_pred),
        })
        print(f"    fold {idx}  MAE={metrics[-1]['MAE']:.4f}  R2={metrics[-1]['R2']:.4f}")

    metrics_df = pd.DataFrame(metrics)
    if not metrics_df.empty:
        print(f"  {label} avg MAE: {metrics_df['MAE'].mean():.4f}")

    final_model = None
    if cfg.retrain_on_full_data and len(df) > 0:
        final_model = RandomForestRegressor(**model_params)
        final_model.fit(df[feat_cols], df[target_col])
        print(f"  {label} model retrained on full data ({len(df):,} rows)")

    return final_model, metrics_df


def _positive_class_probability(model: RandomForestClassifier, x: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(x)
    classes = list(getattr(model, "classes_", []))
    if 1 in classes:
        return proba[:, classes.index(1)]
    if classes == [0]:
        return np.zeros(len(x))
    if classes == [1]:
        return np.ones(len(x))
    return proba[:, -1]


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
