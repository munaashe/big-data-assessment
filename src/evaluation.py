"""Repeated, group-aware evaluation of selected booking-time cancellation features."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
	accuracy_score,
	confusion_matrix,
	f1_score,
	precision_score,
	recall_score,
	roc_auc_score,
)
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

from src.modelling import build_models
from src.preprocessing import (
	DATA_PATH,
	PROJECT_ROOT,
	SELECTED_FEATURES,
	TARGET,
	load_reservations,
)


OUTPUT_DIR = PROJECT_ROOT / "output"
SPLITS = {"80/20": 0.20, "70/30": 0.30, "60/40": 0.40}
SEEDS = (11, 23, 42, 57, 101)
DECISION_THRESHOLDS = (0.30, 0.40, 0.50, 0.60, 0.70)


def _feature_effects(
	model: Pipeline,
	model_name: str,
	split_name: str,
	seed: int,
	feature_set: str,
	numeric_features: set[str],
) -> list[dict[str, object]]:
	"""Aggregate transformed coefficients/importances back to source attributes."""
	preprocessor = model.named_steps["preprocessing"]
	classifier = model.named_steps["classifier"]
	transformed_names = preprocessor.get_feature_names_out()
	if model_name == "RandomForest":
		values = classifier.feature_importances_
		coefficients = [None] * len(values)
	else:
		coefficients = classifier.coef_[0]
		values = abs(coefficients)

	aggregated: dict[str, dict[str, float]] = {}
	for transformed_name, value, coefficient in zip(transformed_names, values, coefficients):
		name = transformed_name.split("__", maxsplit=1)[-1]
		attribute = next(
			(feature for feature in SELECTED_FEATURES if name == feature or name.startswith(f"{feature}_")),
			name,
		)
		entry = aggregated.setdefault(attribute, {"importance": 0.0, "coefficient": 0.0, "levels": 0.0})
		entry["importance"] += float(value)
		entry["levels"] += 1
		if coefficient is not None and attribute in numeric_features:
			entry["coefficient"] += float(coefficient)
	return [
		{
			"split": split_name,
			"seed": seed,
			"model": model_name,
			"feature_set": feature_set,
			"feature": attribute,
			"importance": entry["importance"] if model_name == "RandomForest" else entry["importance"] / entry["levels"],
			"signed_coefficient": (
				entry["coefficient"] / entry["levels"]
				if model_name == "LogisticRegression" and attribute in numeric_features
				else float("nan")
			),
		}
		for attribute, entry in aggregated.items()
	]


def run_evaluation(data_path: str | Path = DATA_PATH, output_dir: str | Path = OUTPUT_DIR) -> pd.DataFrame:
	"""Compare selected inputs and a country/deposit ablation across repeated splits."""
	output = Path(output_dir)
	figures = output / "figures"
	figures.mkdir(parents=True, exist_ok=True)
	data = load_reservations(data_path)
	features = data[SELECTED_FEATURES].copy()
	ablation_features = features.drop(columns=["country", "deposit_type"])
	feature_sets = {
		"selected": features,
		"without_country_deposit": ablation_features,
	}
	target = data[TARGET]
	groups = pd.util.hash_pandas_object(ablation_features, index=False).to_numpy()

	result_rows: list[dict[str, object]] = []
	confusion_rows: list[dict[str, object]] = []
	threshold_rows: list[dict[str, object]] = []
	feature_rows: list[dict[str, object]] = []
	for split_name, test_size in SPLITS.items():
		for seed in SEEDS:
			splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
			train_indices, test_indices = next(splitter.split(features, target, groups=groups))
			y_train, y_test = target.iloc[train_indices], target.iloc[test_indices]
			for feature_set, feature_data in feature_sets.items():
				x_train = feature_data.iloc[train_indices]
				x_test = feature_data.iloc[test_indices]
				numeric_features = x_train.select_dtypes(include="number").columns.tolist()
				categorical_features = [
					column for column in x_train.columns if column not in numeric_features
				]
				models = build_models(numeric_features, categorical_features, random_state=seed)
				for model_name, model in models.items():
					model.fit(x_train, y_train)
					predictions = model.predict(x_test)
					probabilities = model.predict_proba(x_test)[:, list(model.classes_).index(1)]
					result_rows.append(
						{
							"split": split_name,
							"seed": seed,
							"feature_set": feature_set,
							"model": model_name,
							"train_rows": len(x_train),
							"test_rows": len(x_test),
							"test_share": len(x_test) / len(data),
							"test_cancellation_rate": y_test.mean(),
							"accuracy": accuracy_score(y_test, predictions),
							"precision": precision_score(y_test, predictions, zero_division=0),
							"recall": recall_score(y_test, predictions, zero_division=0),
							"f1": f1_score(y_test, predictions, zero_division=0),
							"roc_auc": roc_auc_score(y_test, probabilities),
						}
					)
					tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
					confusion_rows.append(
						{
							"split": split_name,
							"seed": seed,
							"feature_set": feature_set,
							"model": model_name,
							"true_negative": tn,
							"false_positive": fp,
							"false_negative": fn,
							"true_positive": tp,
						}
					)
					feature_rows.extend(
						_feature_effects(
							model,
							model_name,
							split_name,
							seed,
							feature_set,
							set(numeric_features),
						)
					)
					for threshold in DECISION_THRESHOLDS:
						threshold_predictions = (probabilities >= threshold).astype(int)
						threshold_rows.append(
							{
								"split": split_name,
								"seed": seed,
								"feature_set": feature_set,
								"model": model_name,
								"threshold": threshold,
								"precision": precision_score(y_test, threshold_predictions, zero_division=0),
								"recall": recall_score(y_test, threshold_predictions, zero_division=0),
								"f1": f1_score(y_test, threshold_predictions, zero_division=0),
								"false_alerts": int(((threshold_predictions == 1) & (y_test == 0)).sum()),
								"missed_cancellations": int(((threshold_predictions == 0) & (y_test == 1)).sum()),
							}
						)
					print(f"Finished {model_name} ({feature_set}, {split_name}, seed={seed})")

			x_train_full, x_test_full = features.iloc[train_indices], features.iloc[test_indices]
			baseline = DummyClassifier(strategy="most_frequent").fit(x_train_full, y_train)
			baseline_predictions = baseline.predict(x_test_full)
			baseline_probabilities = baseline.predict_proba(x_test_full)[:, list(baseline.classes_).index(1)]
			result_rows.append(
				{
					"split": split_name,
					"seed": seed,
					"feature_set": "selected",
					"model": "MajorityBaseline",
					"train_rows": len(x_train_full),
					"test_rows": len(x_test_full),
					"test_share": len(x_test_full) / len(data),
					"test_cancellation_rate": y_test.mean(),
					"accuracy": accuracy_score(y_test, baseline_predictions),
					"precision": precision_score(y_test, baseline_predictions, zero_division=0),
					"recall": recall_score(y_test, baseline_predictions, zero_division=0),
					"f1": f1_score(y_test, baseline_predictions, zero_division=0),
					"roc_auc": roc_auc_score(y_test, baseline_probabilities),
				}
			)
			tn, fp, fn, tp = confusion_matrix(y_test, baseline_predictions, labels=[0, 1]).ravel()
			confusion_rows.append(
				{
					"split": split_name,
					"seed": seed,
					"feature_set": "selected",
					"model": "MajorityBaseline",
					"true_negative": tn,
					"false_positive": fp,
					"false_negative": fn,
					"true_positive": tp,
				}
			)

	runs = pd.DataFrame(result_rows)
	runs.to_csv(output / "model_performance_runs.csv", index=False)
	metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc", "test_share", "test_cancellation_rate", "train_rows", "test_rows"]
	results = runs.groupby(["split", "feature_set", "model"])[metric_names].agg(["mean", "std"])
	results.columns = [f"{metric}_{statistic}" for metric, statistic in results.columns]
	results = results.reset_index()
	results.to_csv(output / "model_performance.csv", index=False)
	ablation = results[results["model"] != "MajorityBaseline"].pivot(
		index=["split", "model"],
		columns="feature_set",
		values=["f1_mean", "roc_auc_mean", "accuracy_mean"],
	)
	ablation.columns = [f"{metric}_{feature_set}" for metric, feature_set in ablation.columns]
	for metric in ["f1_mean", "roc_auc_mean", "accuracy_mean"]:
		ablation[f"{metric}_change_without_country_deposit"] = (
			ablation[f"{metric}_without_country_deposit"] - ablation[f"{metric}_selected"]
		)
	ablation.reset_index().to_csv(output / "ablation_comparison.csv", index=False)
	pd.DataFrame(confusion_rows).to_csv(output / "confusion_matrices.csv", index=False)

	feature_effects = pd.DataFrame(feature_rows)
	feature_effects = feature_effects.groupby(["feature_set", "model", "feature"])[["importance", "signed_coefficient"]].agg(["mean", "std"])
	feature_effects.columns = [f"{metric}_{statistic}" for metric, statistic in feature_effects.columns]
	feature_effects.reset_index().sort_values(["model", "importance_mean"], ascending=[True, False]).to_csv(
		output / "feature_importance.csv", index=False
	)

	thresholds = pd.DataFrame(threshold_rows)
	threshold_summary = thresholds.groupby(["split", "feature_set", "model", "threshold"])[
		["precision", "recall", "f1", "false_alerts", "missed_cancellations"]
	].agg(["mean", "std"])
	threshold_summary.columns = [f"{metric}_{statistic}" for metric, statistic in threshold_summary.columns]
	threshold_summary.reset_index().to_csv(output / "threshold_tradeoff.csv", index=False)

	fig, axis = plt.subplots(figsize=(9, 5))
	for (feature_set, model_name), model_results in results.groupby(["feature_set", "model"]):
		axis.errorbar(
			model_results["split"],
			model_results["f1_mean"],
			yerr=model_results["f1_std"],
			marker="o",
			capsize=3,
			label=f"{model_name}: {feature_set}",
		)
	axis.set(xlabel="Train/test split", ylabel="F1 score", title="Cancellation classifier performance")
	axis.set_ylim(0, 1)
	axis.legend()
	fig.tight_layout()
	fig.savefig(figures / "model_f1_comparison.png", dpi=160)
	plt.close(fig)

	print("\nModel performance:")
	print(results[["split", "feature_set", "model", "f1_mean", "f1_std", "roc_auc_mean", "accuracy_mean"]].to_string(
		index=False, float_format=lambda value: f"{value:.3f}"
	))
	print(f"\nEvaluation outputs saved to {output}")
	return results


if __name__ == "__main__":
	run_evaluation()
