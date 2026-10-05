"""Load and prepare the hotel reservations data for analysis and modelling."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "hotel-reservations.csv"
TARGET = "is_canceled"

# These fields are unavailable at the initial booking decision or encode identifiers.
EXCLUDED_FEATURES = {
	TARGET,
	"reservation_status",
	"reservation_status_date",
	"assigned_room_type",
	"agent",
	"company",
	"required_car_parking_spaces",
	"booking_changes",
	"days_in_waiting_list",
}

SELECTED_FEATURES = [
	"hotel",
	"lead_time",
	"arrival_date_month",
	"meal",
	"country",
	"market_segment",
	"distribution_channel",
	"previous_cancellations",
	"reserved_room_type",
	"deposit_type",
	"customer_type",
	"adr",
	"total_of_special_requests",
]


def load_reservations(path: str | Path = DATA_PATH) -> pd.DataFrame:
	"""Apply documented cleaning while retaining valid reservation rows."""
	data = pd.read_csv(path)
	required = {TARGET, "country", "arrival_date_month", "lead_time"}
	missing = required.difference(data.columns)
	if missing:
		raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

	data, _ = clean_reservations(data)
	data[TARGET] = pd.to_numeric(data[TARGET], errors="raise").astype("int8")
	if not data[TARGET].isin([0, 1]).all():
		raise ValueError(f"{TARGET} must contain only 0 and 1")
	return data


def clean_reservations(
	data: pd.DataFrame,
	cap_adr: bool = False,
) -> tuple[pd.DataFrame, dict[str, object]]:
	"""Clean data and return counts/thresholds for an auditable preprocessing summary."""
	clean = data.copy()
	rows_before = len(clean)
	cn_rows = int(clean["country"].eq("CN").sum())
	undefined_meals = int(clean["meal"].eq("Undefined").sum())
	negative_adr = int(pd.to_numeric(clean["adr"], errors="coerce").lt(0).sum())
	clean["country"] = clean["country"].replace({"CN": "CHN"})
	clean["meal"] = clean["meal"].replace({"Undefined": "SC"})
	clean["adr"] = pd.to_numeric(clean["adr"], errors="coerce")
	clean["children"] = pd.to_numeric(clean["children"], errors="coerce")
	clean = clean.loc[clean["adr"].ge(0)].copy()
	guest_count = clean[["adults", "children", "babies"]].fillna(0).sum(axis=1)
	zero_guest = int(guest_count.eq(0).sum())
	clean = clean.loc[guest_count.gt(0)].copy()
	adr_cap = float(clean["adr"].quantile(0.995)) if cap_adr else None
	adr_capped = int(clean["adr"].gt(adr_cap).sum()) if adr_cap is not None else 0
	if adr_cap is not None:
		clean["adr"] = clean["adr"].clip(upper=adr_cap)
	clean[TARGET] = pd.to_numeric(clean[TARGET], errors="raise").astype("int8")
	if not clean[TARGET].isin([0, 1]).all():
		raise ValueError(f"{TARGET} must contain only 0 and 1")
	return clean, {
		"rows_before": rows_before,
		"rows_after": len(clean),
		"negative_adr_rows_removed": negative_adr,
		"zero_guest_rows_removed": zero_guest,
		"country_cn_rows_merged_to_chn": cn_rows,
		"undefined_meal_values_merged_to_sc": undefined_meals,
		"adr_cap_quantile": 0.995,
		"adr_cap_value": adr_cap,
		"adr_values_capped": adr_capped,
		"missing_values_after_cleaning": clean.isna().sum().loc[lambda count: count.gt(0)].to_dict(),
	}


def get_feature_columns(data: pd.DataFrame) -> list[str]:
	"""Return booking-time predictors, excluding identifiers and outcome leakage."""
	return [column for column in data.columns if column not in EXCLUDED_FEATURES]


def get_selected_features(data: pd.DataFrame) -> list[str]:
	"""Return the assessment's preselected, interpretable reservation-time features."""
	missing = set(SELECTED_FEATURES).difference(data.columns)
	if missing:
		raise ValueError(f"Dataset is missing selected features: {sorted(missing)}")
	return SELECTED_FEATURES.copy()
