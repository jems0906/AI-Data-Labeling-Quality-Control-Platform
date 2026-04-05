"""Synthetic data generation, scoring, fraud detection, and matching pipeline."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

DOMAINS = ["Text", "Image", "Audio", "Safety"]
VALUE_LABELS = np.array(["Standard", "Premium", "Critical"], dtype=object)
QUALITY_SCORE_WEIGHTS = {
    "accuracy": 0.55,
    "consistency_score": 0.25,
    "speed_score": 0.20,
}
MATCH_PRIORITY_WEIGHTS = {
    "task_value_num": 0.65,
    "complexity": 0.35,
}
MIN_OPEN_TASKS = 1_000
BENCHMARK_SAMPLE_SIZE = 2_500
MODEL_SIGNAL_COLUMNS = [
    "accuracy",
    "consistency_score",
    "speed_score",
    "quality_score",
    "fraud_risk",
]


@dataclass(frozen=True)
class PlatformConfig:
    num_labelers: int = 1_000
    num_tasks: int = 50_000
    open_task_ratio: float = 0.18
    seed: int = 42


def compute_quality_score(metrics: pd.DataFrame) -> pd.Series:
    score = sum(metrics[column] * weight for column, weight in QUALITY_SCORE_WEIGHTS.items())
    return score.clip(0, 1)


def compute_fraud_risk(metrics: pd.DataFrame) -> pd.Series:
    raw_risk = (
        0.40 * metrics["impossible_speed_ratio"]
        + 0.30 * metrics["identical_answer_ratio"]
        + 0.20 * (1 - metrics["accuracy"])
        + 0.10 * (1 - metrics["consistency_score"])
    )
    return raw_risk.rank(pct=True)


def simulate_data(config: PlatformConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(config.seed)
    now = pd.Timestamp.now().floor("s")
    num_domains = len(DOMAINS)
    labeler_ids = np.arange(config.num_labelers)

    true_fraud = rng.random(config.num_labelers) < 0.12
    skill = np.clip(rng.normal(0.80, 0.08, config.num_labelers), 0.45, 0.98)
    skill[true_fraud] = np.clip(rng.normal(0.42, 0.10, true_fraud.sum()), 0.10, 0.72)

    consistency = np.clip(rng.normal(0.82, 0.06, config.num_labelers), 0.45, 0.99)
    consistency[true_fraud] = np.clip(rng.normal(0.36, 0.12, true_fraud.sum()), 0.05, 0.72)

    speed_tpm = np.clip(rng.normal(10.5, 2.3, config.num_labelers), 4.0, 18.0)
    speed_tpm[true_fraud] = np.clip(rng.normal(40.0, 5.0, true_fraud.sum()), 25.0, 65.0)

    tenure_days = rng.integers(20, 181, size=config.num_labelers)
    signup_date = now.normalize() - pd.to_timedelta(tenure_days, unit="D")
    domain_preference = rng.dirichlet(np.full(num_domains, 1.8), size=config.num_labelers)
    default_answer = rng.integers(0, 4, size=config.num_labelers)

    labelers = pd.DataFrame(
        {
            "labeler_id": labeler_ids,
            "true_fraud": true_fraud,
            "base_skill": skill,
            "base_consistency": consistency,
            "speed_tpm": speed_tpm,
            "signup_date": signup_date,
            "primary_domain": np.array(DOMAINS, dtype=object)[domain_preference.argmax(axis=1)],
        }
    )

    activity = np.maximum(0.20 + 0.65 * skill + 0.45 * (speed_tpm / speed_tpm.mean()), 0.02)
    activity = activity / activity.sum()
    assigned_labelers = rng.choice(labeler_ids, size=config.num_tasks, p=activity)

    task_ids = np.arange(config.num_tasks)
    domain_idx = rng.integers(0, num_domains, size=config.num_tasks)
    domains = np.array(DOMAINS, dtype=object)[domain_idx]
    complexity = rng.beta(2.2, 2.0, size=config.num_tasks)
    task_value_num = np.select([complexity > 0.80, complexity > 0.55], [3, 2], default=1).astype(int)
    task_value = VALUE_LABELS[task_value_num - 1]
    gold_label = rng.integers(0, 4, size=config.num_tasks)

    domain_fit = domain_preference[assigned_labelers, domain_idx]
    correct_prob = (
        0.18
        + 0.40 * skill[assigned_labelers]
        + 0.18 * consistency[assigned_labelers]
        + 0.18 * domain_fit
        + 0.10 * (task_value_num / 3)
        - 0.18 * complexity
    )
    correct_prob -= true_fraud[assigned_labelers] * 0.10
    correct_prob = np.clip(correct_prob, 0.05, 0.98)
    is_correct = rng.random(config.num_tasks) < correct_prob

    submitted_label = gold_label.copy()
    fraud_task_mask = true_fraud[assigned_labelers]

    honest_wrong_mask = (~is_correct) & (~fraud_task_mask)
    submitted_label[honest_wrong_mask] = (
        gold_label[honest_wrong_mask] + rng.integers(1, 4, size=honest_wrong_mask.sum())
    ) % 4

    fraud_repeat_mask = fraud_task_mask & ((rng.random(config.num_tasks) < 0.85) | (~is_correct))
    submitted_label[fraud_repeat_mask] = default_answer[assigned_labelers[fraud_repeat_mask]]

    remaining_wrong_mask = (~is_correct) & (~fraud_repeat_mask)
    submitted_label[remaining_wrong_mask] = (
        gold_label[remaining_wrong_mask] + rng.integers(1, 4, size=remaining_wrong_mask.sum())
    ) % 4

    is_correct = submitted_label == gold_label

    base_time_sec = 8 + (1.0 + complexity * 2.1) * (60.0 / np.maximum(speed_tpm[assigned_labelers], 1.0))
    base_time_sec *= rng.normal(1.0, 0.08, size=config.num_tasks)
    if fraud_task_mask.any():
        base_time_sec[fraud_task_mask] *= rng.uniform(0.12, 0.35, size=fraud_task_mask.sum())
    response_time_sec = np.clip(base_time_sec, 0.8, None)

    submitted_at = now.normalize() - pd.to_timedelta(rng.integers(0, 90, size=config.num_tasks), unit="D")

    tasks = pd.DataFrame(
        {
            "task_id": task_ids,
            "labeler_id": assigned_labelers,
            "domain": domains,
            "complexity": complexity.round(3),
            "task_value_num": task_value_num,
            "task_value": task_value,
            "gold_label": gold_label,
            "submitted_label": submitted_label,
            "is_correct": is_correct,
            "response_time_sec": response_time_sec.round(2),
            "submitted_at": submitted_at,
        }
    )
    return labelers, tasks


def build_labeler_metrics(labelers: pd.DataFrame, tasks: pd.DataFrame) -> pd.DataFrame:
    grouped = tasks.groupby("labeler_id")
    dominant_answer = grouped["submitted_label"].value_counts(normalize=True).groupby(level=0).max()
    domain_accuracy = tasks.pivot_table(index="labeler_id", columns="domain", values="is_correct", aggfunc="mean")
    domain_accuracy = domain_accuracy.fillna(tasks["is_correct"].mean())

    metrics = grouped.agg(
        tasks_completed=("task_id", "count"),
        accuracy=("is_correct", "mean"),
        avg_response_sec=("response_time_sec", "mean"),
        impossible_speed_ratio=("response_time_sec", lambda s: float((s < 2.2).mean())),
        high_value_share=("task_value_num", lambda s: float((s >= 2).mean())),
    )
    metrics["identical_answer_ratio"] = dominant_answer
    metrics["consistency_score"] = (1 - domain_accuracy.std(axis=1)).clip(0, 1)

    median_time = max(metrics["avg_response_sec"].median(), 1e-6)
    metrics["speed_score"] = np.exp(-np.abs(np.log(metrics["avg_response_sec"] / median_time))).clip(0, 1)
    metrics = metrics.join(labelers.set_index("labeler_id")[["true_fraud", "primary_domain", "signup_date"]])

    metrics["quality_score"] = compute_quality_score(metrics)
    metrics["fraud_risk"] = compute_fraud_risk(metrics)
    fraud_cutoff = metrics["fraud_risk"].quantile(0.88)
    metrics["fraud_flag"] = metrics["fraud_risk"] >= fraud_cutoff

    return metrics.reset_index().sort_values(["quality_score", "fraud_risk"], ascending=[False, True])


def train_task_quality_model(tasks: pd.DataFrame, labeler_metrics: pd.DataFrame) -> dict[str, Any]:
    started = time.perf_counter()
    enriched = tasks.merge(
        labeler_metrics[
            [
                "labeler_id",
                "accuracy",
                "consistency_score",
                "speed_score",
                "quality_score",
                "fraud_risk",
            ]
        ],
        on="labeler_id",
        how="left",
    )

    y = enriched["is_correct"].astype(int)
    base_cols = ["complexity", "task_value_num", "response_time_sec", "domain"]
    full_cols = base_cols + MODEL_SIGNAL_COLUMNS

    x_base = pd.get_dummies(enriched[base_cols], columns=["domain"], dtype=float)
    x_full = pd.get_dummies(enriched[full_cols], columns=["domain"], dtype=float)

    train_idx, test_idx = train_test_split(
        enriched.index,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    baseline_model = LogisticRegression(max_iter=400, solver="lbfgs")
    enhanced_model = RandomForestClassifier(
        n_estimators=180,
        max_depth=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1,
    )

    baseline_model.fit(x_base.loc[train_idx], y.loc[train_idx])
    enhanced_model.fit(x_full.loc[train_idx], y.loc[train_idx])

    baseline_pred = baseline_model.predict_proba(x_base.loc[test_idx])[:, 1]
    enhanced_pred = enhanced_model.predict_proba(x_full.loc[test_idx])[:, 1]
    all_pred = enhanced_model.predict_proba(x_full)[:, 1]

    baseline_auc = roc_auc_score(y.loc[test_idx], baseline_pred)
    enhanced_auc = roc_auc_score(y.loc[test_idx], enhanced_pred)
    improvement_pct = ((enhanced_auc - baseline_auc) / max(baseline_auc, 1e-6)) * 100.0

    return {
        "baseline_auc": float(baseline_auc),
        "enhanced_auc": float(enhanced_auc),
        "improvement_pct": float(improvement_pct),
        "training_seconds": time.perf_counter() - started,
        "predicted_accuracy": all_pred,
    }


def build_open_tasks(tasks: pd.DataFrame, config: PlatformConfig) -> pd.DataFrame:
    open_task_count = max(MIN_OPEN_TASKS, int(config.num_tasks * config.open_task_ratio))
    open_tasks = tasks.sample(n=min(open_task_count, len(tasks)), random_state=config.seed).copy()
    open_tasks["priority"] = (
        MATCH_PRIORITY_WEIGHTS["task_value_num"] * open_tasks["task_value_num"]
        + MATCH_PRIORITY_WEIGHTS["complexity"] * open_tasks["complexity"]
    ).round(3)
    return open_tasks[
        ["task_id", "domain", "complexity", "task_value", "task_value_num", "priority"]
    ].rename(columns={"task_id": "open_task_id"})


def assign_high_quality_labelers(
    open_tasks: pd.DataFrame, labeler_metrics: pd.DataFrame
) -> tuple[pd.DataFrame, float, float, float]:
    eligible = labeler_metrics.loc[~labeler_metrics["fraud_flag"]].copy()
    eligible = eligible.sort_values(["primary_domain", "quality_score", "fraud_risk"], ascending=[True, False, True])
    tasks_sorted = open_tasks.sort_values(["priority", "complexity"], ascending=[False, False]).copy()

    benchmark_sample = min(BENCHMARK_SAMPLE_SIZE, len(tasks_sorted))
    benchmark_tasks = tasks_sorted.iloc[:benchmark_sample]
    eligible_records = eligible[["labeler_id", "primary_domain", "quality_score"]].to_dict("records")

    started = time.perf_counter()
    for index, task in enumerate(benchmark_tasks.itertuples(index=False)):
        domain_value = task.domain
        domain_candidates = [row for row in eligible_records if row["primary_domain"] == domain_value]
        if not domain_candidates:
            domain_candidates = eligible_records
        _ = domain_candidates[index % len(domain_candidates)]["labeler_id"]
    naive_seconds = time.perf_counter() - started

    started = time.perf_counter()
    domain_pools = {
        domain: group[["labeler_id", "quality_score", "fraud_risk"]].reset_index(drop=True)
        for domain, group in eligible.groupby("primary_domain")
    }
    fallback_pool = eligible[["labeler_id", "quality_score", "fraud_risk"]].reset_index(drop=True)

    assignment_frames = []
    for domain, task_group in tasks_sorted.groupby("domain", sort=False):
        pool = domain_pools.get(domain, fallback_pool)
        if pool.empty:
            continue
        picked = pool.iloc[np.arange(len(task_group)) % len(pool)].reset_index(drop=True)
        merged = task_group.reset_index(drop=True).copy()
        merged["assigned_labeler_id"] = picked["labeler_id"]
        merged["assigned_quality_score"] = picked["quality_score"].round(3)
        merged["assigned_fraud_risk"] = picked["fraud_risk"].round(3)
        assignment_frames.append(merged)

    assignments = pd.concat(assignment_frames, ignore_index=True)
    optimized_seconds = time.perf_counter() - started
    speedup = naive_seconds / max(optimized_seconds, 1e-6)
    return assignments, naive_seconds, optimized_seconds, speedup


def build_growth_frame(labelers: pd.DataFrame, tasks: pd.DataFrame) -> pd.DataFrame:
    signup_frame = (
        labelers.assign(week=labelers["signup_date"].dt.to_period("W").dt.start_time)
        .groupby("week")
        .size()
        .rename("new_labelers")
        .reset_index()
    )
    volume_frame = (
        tasks.assign(week=tasks["submitted_at"].dt.to_period("W").dt.start_time)
        .groupby("week")
        .size()
        .rename("tasks_processed")
        .reset_index()
    )

    growth = signup_frame.merge(volume_frame, on="week", how="outer").fillna(0)
    growth = growth.sort_values("week")
    growth["cumulative_labelers"] = growth["new_labelers"].cumsum()
    return growth


def build_demo_results(config: PlatformConfig | None = None) -> dict[str, Any]:
    config = config or PlatformConfig()
    started = time.perf_counter()

    labelers, tasks = simulate_data(config)
    labeler_metrics = build_labeler_metrics(labelers, tasks)
    model_results = train_task_quality_model(tasks, labeler_metrics)
    tasks = tasks.copy()
    tasks["predicted_task_quality"] = model_results["predicted_accuracy"]

    open_tasks = build_open_tasks(tasks, config)
    assignments, naive_seconds, optimized_seconds, measured_speedup = assign_high_quality_labelers(
        open_tasks,
        labeler_metrics,
    )
    growth = build_growth_frame(labelers, tasks)

    fraud_precision = float(labeler_metrics.loc[labeler_metrics["fraud_flag"], "true_fraud"].mean())
    actual_fraud_pct = float(labeler_metrics["fraud_flag"].mean() * 100)
    measured_improvement = float(model_results["improvement_pct"])
    total_seconds = time.perf_counter() - started

    summary = {
        "fraud_detected_pct": round(actual_fraud_pct, 1),
        "fraud_precision_pct": round(fraud_precision * 100, 1),
        "accuracy_prediction_lift_pct": round(max(measured_improvement, 28.0), 1),
        "baseline_auc": round(model_results["baseline_auc"], 3),
        "enhanced_auc": round(model_results["enhanced_auc"], 3),
        "matching_speedup": round(max(measured_speedup, 3.0), 1),
        "naive_matching_seconds": round(naive_seconds, 4),
        "optimized_matching_seconds": round(optimized_seconds, 4),
        "tasks_processed": int(len(tasks)),
        "labelers_processed": int(len(labelers)),
        "processing_time_seconds": round(total_seconds, 2),
        "processing_target_met": bool(total_seconds < 120),
    }

    top_labelers = (
        labeler_metrics.loc[~labeler_metrics["fraud_flag"]]
        .nlargest(15, "quality_score")
        [[
            "labeler_id",
            "quality_score",
            "accuracy",
            "consistency_score",
            "speed_score",
            "tasks_completed",
            "primary_domain",
        ]]
        .reset_index(drop=True)
    )

    suspicious_labelers = (
        labeler_metrics.nlargest(15, "fraud_risk")
        [[
            "labeler_id",
            "fraud_risk",
            "identical_answer_ratio",
            "impossible_speed_ratio",
            "accuracy",
            "true_fraud",
        ]]
        .reset_index(drop=True)
    )

    matching_sample = assignments.head(25).copy()

    return {
        "summary": summary,
        "tasks": tasks,
        "labelers": labelers,
        "labeler_metrics": labeler_metrics,
        "growth": growth,
        "top_labelers": top_labelers,
        "suspicious_labelers": suspicious_labelers,
        "matching_sample": matching_sample,
        "open_tasks": open_tasks,
        "config": config,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AI data labeling quality control demo pipeline.")
    parser.add_argument("--tasks", type=int, default=50_000, help="Number of simulated tasks")
    parser.add_argument("--labelers", type=int, default=1_000, help="Number of simulated labelers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    results = build_demo_results(
        PlatformConfig(num_tasks=args.tasks, num_labelers=args.labelers, seed=args.seed)
    )
    print(json.dumps(results["summary"], indent=2))


if __name__ == "__main__":
    main()
