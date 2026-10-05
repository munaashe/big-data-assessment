"""Explore cancellation patterns and save tables and figures for the report."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import roc_auc_score

from src.preprocessing import (
	DATA_PATH,
	EXCLUDED_FEATURES,
	PROJECT_ROOT,
	SELECTED_FEATURES,
	TARGET,
	clean_reservations,
)


OUTPUT_DIR = PROJECT_ROOT / "output"
MONTH_ORDER = [
	"January", "February", "March", "April", "May", "June",
	"July", "August", "September", "October", "November", "December",
]
MIN_COUNTRY_BOOKINGS = 100
MIN_GROUP_BOOKINGS = 500
MIN_COUNTRY_MONTH_BOOKINGS = 50
WILSON_Z = 1.96


def add_wilson_intervals(table: pd.DataFrame) -> pd.DataFrame:
	"""Add two-sided 95% Wilson score intervals for binomial rates."""
	result = table.copy()
	n = result["bookings"].to_numpy(dtype=float)
	rate = result["cancellation_rate"].to_numpy(dtype=float)
	z_squared = WILSON_Z**2
	denominator = 1 + z_squared / n
	center = (rate + z_squared / (2 * n)) / denominator
	half_width = WILSON_Z * np.sqrt(rate * (1 - rate) / n + z_squared / (4 * n**2)) / denominator
	result["ci_95_lower"] = center - half_width
	result["ci_95_upper"] = center + half_width
	return result


def country_cancellation_tables(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
	"""Return country rates and supported country-month cells for the top six countries."""
	known = data.dropna(subset=["country"]).copy()
	overall = (
		known.groupby("country", observed=True)[TARGET]
		.agg(bookings="size", cancellation_rate="mean")
		.query("bookings >= @MIN_COUNTRY_BOOKINGS")
		.reset_index()
	)
	overall = add_wilson_intervals(overall).sort_values(
		["cancellation_rate", "bookings"], ascending=[False, False]
	)
	monthly = (
		known.groupby(["country", "arrival_date_month"], observed=True)[TARGET]
		.agg(bookings="size", cancellation_rate="mean")
		.reset_index()
	)
	monthly = monthly[monthly["bookings"] >= MIN_COUNTRY_MONTH_BOOKINGS].copy()
	monthly = add_wilson_intervals(monthly)
	ranked_countries = (
		monthly.groupby("country", observed=True)["ci_95_lower"]
		.max()
		.sort_values(ascending=False)
	)
	top_countries = ranked_countries.head(6).index
	monthly = monthly[monthly["country"].isin(top_countries)].copy()
	monthly["arrival_date_month"] = pd.Categorical(
		monthly["arrival_date_month"], categories=MONTH_ORDER, ordered=True
	)
	monthly = monthly.sort_values(["ci_95_lower", "bookings"], ascending=[False, False])
	return overall, monthly


def attribute_cancellation_rates(data: pd.DataFrame) -> pd.DataFrame:
	"""Compare cancellation rates across useful categorical levels and numeric bins."""
	rows: list[pd.DataFrame] = []
	baseline = data[TARGET].mean()
	for column in data.columns:
		if column in {TARGET, "reservation_status", "reservation_status_date", "agent", "company"}:
			continue
		values = data[column]
		if pd.api.types.is_numeric_dtype(values):
			groups = pd.qcut(values, q=5, duplicates="drop").astype("string").fillna("Missing")
		else:
			groups = values.fillna("Missing").astype("string")
		grouped = data.assign(_group=groups).groupby("_group", observed=True)[TARGET]
		summary = grouped.agg(bookings="size", cancellation_rate="mean").reset_index()
		summary = summary[summary["bookings"] >= MIN_GROUP_BOOKINGS].copy()
		if summary.empty:
			continue
		summary.insert(0, "attribute", column)
		summary = summary.rename(columns={"_group": "value_group"})
		summary["value_group"] = summary["value_group"].astype(str)
		summary["risk_difference"] = summary["cancellation_rate"] - baseline
		summary["model_eligible"] = column not in EXCLUDED_FEATURES
		rows.append(summary)

	if not rows:
		return pd.DataFrame(
			columns=["attribute", "value_group", "bookings", "cancellation_rate", "risk_difference"]
		)
	return pd.concat(rows, ignore_index=True).sort_values(
		"risk_difference", ascending=False
	)


def attribute_associations(data: pd.DataFrame) -> pd.DataFrame:
	"""Estimate single-feature AUC for numeric fields and Cramer's V for categories."""
	rows: list[dict[str, object]] = []
	for column in data.columns:
		if column in {TARGET, "reservation_status", "reservation_status_date", "agent", "company"}:
			continue
		values = data[column]
		if pd.api.types.is_numeric_dtype(values):
			valid = values.notna()
			auc = roc_auc_score(data.loc[valid, TARGET], values.loc[valid]) if values.loc[valid].nunique() > 1 else np.nan
			rows.append({
				"attribute": column,
				"attribute_type": "numeric",
				"association_measure": "single_feature_roc_auc",
				"association_strength": 2 * abs(auc - 0.5) if pd.notna(auc) else np.nan,
				"raw_auc": auc,
				"p_value": np.nan,
				"model_eligible": column not in EXCLUDED_FEATURES,
			})
			continue

		categories = values.fillna("Missing").astype(str)
		contingency = pd.crosstab(categories, data[TARGET])
		if min(contingency.shape) < 2:
			continue
		chi_square, p_value, _, _ = chi2_contingency(contingency, correction=False)
		denominator = len(data) * min(contingency.shape[0] - 1, contingency.shape[1] - 1)
		cramers_v = np.sqrt(chi_square / denominator) if denominator else np.nan
		rows.append({
			"attribute": column,
			"attribute_type": "categorical",
			"association_measure": "cramers_v",
			"association_strength": cramers_v,
			"raw_auc": np.nan,
			"p_value": p_value,
			"model_eligible": column not in EXCLUDED_FEATURES,
		})
	return pd.DataFrame(rows).sort_values("association_strength", ascending=False)


def run_analysis(data_path: str | Path = DATA_PATH, output_dir: str | Path = OUTPUT_DIR) -> None:
	"""Generate the Section 1 analysis tables and report-ready charts."""
	output = Path(output_dir)
	figures = output / "figures"
	figures.mkdir(parents=True, exist_ok=True)
	raw = pd.read_csv(data_path)
	data, cleaning_summary = clean_reservations(raw, cap_adr=True)

	country_overall, country_monthly = country_cancellation_tables(data)
	attributes = attribute_cancellation_rates(data)
	associations = attribute_associations(data)
	lead_time = attributes[attributes["attribute"] == "lead_time"].copy()
	lead_time_auc = associations.loc[associations["attribute"].eq("lead_time"), "raw_auc"]

	country_overall.to_csv(output / "country_cancellation_rates.csv", index=False)
	country_monthly.to_csv(output / "top_six_country_month_rates.csv", index=False)
	attributes.to_csv(output / "attribute_cancellation_rates.csv", index=False)
	associations.to_csv(output / "attribute_association.csv", index=False)
	lead_time.to_csv(output / "lead_time_cancellation_rates.csv", index=False)

	top = country_overall.head(6).sort_values("cancellation_rate")
	fig, axis = plt.subplots(figsize=(9, 5))
	axis.barh(top["country"], top["cancellation_rate"], color="#177e89")
	axis.set(xlabel="Cancellation rate", ylabel="Country", title="Countries with highest cancellation rates (100+ bookings)")
	axis.set_xlim(0, 1)
	fig.tight_layout()
	fig.savefig(figures / "top_countries_cancellation_rate.png", dpi=160)
	plt.close(fig)

	heatmap = country_monthly.pivot(
		index="country", columns="arrival_date_month", values="cancellation_rate"
	).reindex(columns=MONTH_ORDER)
	fig, axis = plt.subplots(figsize=(12, 5))
	image = axis.imshow(heatmap.to_numpy(), aspect="auto", vmin=0, vmax=1, cmap="YlOrRd")
	axis.set_xticks(range(len(heatmap.columns)), heatmap.columns, rotation=35, ha="right")
	axis.set_yticks(range(len(heatmap.index)), heatmap.index)
	axis.set(xlabel="Arrival month", ylabel="Country", title="Cancellation rate by country and month (n ≥ 50)")
	fig.colorbar(image, ax=axis, label="Cancellation rate")
	fig.tight_layout()
	fig.savefig(figures / "country_month_cancellation_heatmap.png", dpi=160)
	plt.close(fig)

	lead_time["value_group"] = lead_time["value_group"].astype(str)
	fig, axis = plt.subplots(figsize=(9, 5))
	axis.bar(lead_time["value_group"], lead_time["cancellation_rate"], color="#e07a5f")
	axis.set(xlabel="Lead-time quintile (days)", ylabel="Cancellation rate", title="Cancellation rate by lead time")
	axis.tick_params(axis="x", labelrotation=25)
	fig.tight_layout()
	fig.savefig(figures / "lead_time_cancellation_rate.png", dpi=160)
	plt.close(fig)

	non_refund = data["deposit_type"].eq("Non Refund")
	non_refund_country_share = (
		data.loc[non_refund, "country"].value_counts(normalize=True).get("PRT", 0.0)
	)
	decision_areas = [
		{
			"area": "Deposit and market-segment interaction",
			"evidence": f"Non-refundable bookings cancel at {data.loc[non_refund, TARGET].mean():.1%}; {non_refund_country_share:.1%} are Portuguese.",
			"recommendation": "Audit deposit policy by country and market segment before treating deposit type as an independent cause.",
		},
		{
			"area": "Long-lead-time cancellations",
			"evidence": f"Single-feature lead-time AUC is {lead_time_auc.iloc[0]:.3f} after orienting toward the stronger class association.",
			"recommendation": "Evaluate targeted reconfirmation or resale preparation for long-lead bookings, using measured precision and recall to set an operational threshold.",
		},
		{
			"area": "Reservation-time risk triage",
			"evidence": "The selected booking-time features are evaluated against a majority-class baseline and with repeated group-aware splits.",
			"recommendation": "Pilot the best calibrated model as a review aid; monitor false positives and missed cancellations before changing inventory policy.",
		},
	]

	summary = {
		"rows_analyzed": len(data),
		"cleaning": cleaning_summary,
		"cancellation_rate": float(data[TARGET].mean()),
		"countries_in_top_six": country_monthly["country"].drop_duplicates().tolist(),
		"country_month_min_bookings": MIN_COUNTRY_MONTH_BOOKINGS,
		"lead_time_single_feature_auc_strength": float(lead_time_auc.iloc[0]),
		"selected_model_features": SELECTED_FEATURES,
		"highest_risk_groups": attributes.head(10).to_dict(orient="records"),
		"strongest_associations": associations.head(10).to_dict(orient="records"),
		"lead_time_quintiles": lead_time.to_dict(orient="records"),
		"non_refundable_booking_count": int(non_refund.sum()),
		"non_refundable_cancellation_rate": float(data.loc[non_refund, TARGET].mean()),
		"non_refundable_portuguese_share": float(non_refund_country_share),
		"decision_areas": decision_areas,
	}
	(output / "analysis_summary.json").write_text(json.dumps(summary, indent=2, default=str))
	print(f"Analysis outputs saved to {output}")


if __name__ == "__main__":
	run_analysis()
