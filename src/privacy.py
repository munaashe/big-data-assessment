"""Measure record uniqueness under progressively coarser quasi-identifiers."""

import json
from pathlib import Path

import country_converter as coco
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.preprocessing import DATA_PATH, PROJECT_ROOT, load_reservations


OUTPUT_DIR = PROJECT_ROOT / "output"
MONTH_NUMBER = {
	"January": "01", "February": "02", "March": "03", "April": "04",
	"May": "05", "June": "06", "July": "07", "August": "08",
	"September": "09", "October": "10", "November": "11", "December": "12",
}


def _party_size_bucket(party_size: int) -> str:
	if party_size <= 1:
		return "1"
	if party_size == 2:
		return "2"
	if party_size <= 4:
		return "3-4"
	return "5+"


def _k_statistics(data: pd.DataFrame, scenario: str, columns: list[str]) -> dict[str, object]:
	"""Return row-weighted k-anonymity risk measures for one quasi-identifier set."""
	group_sizes = data.groupby(columns, dropna=False, observed=True)[columns[0]].transform("size")
	return {
		"scenario": scenario,
		"quasi_identifiers": ", ".join(columns),
		"records": len(data),
		"unique_equivalence_classes": int(data[columns].drop_duplicates().shape[0]),
		"minimum_k": int(group_sizes.min()),
		"median_k": float(group_sizes.median()),
		"share_k_1": float(group_sizes.eq(1).mean()),
		"share_k_below_5": float(group_sizes.lt(5).mean()),
	}


def run_privacy_analysis(data_path: str | Path = DATA_PATH, output_dir: str | Path = OUTPUT_DIR) -> pd.DataFrame:
	"""Compare exact-date and generalized quasi-identifiers without exporting rows."""
	output = Path(output_dir)
	figures = output / "figures"
	figures.mkdir(parents=True, exist_ok=True)
	data = load_reservations(data_path).copy()
	data["country_qi"] = data["country"].fillna("Unknown").astype(str)
	data["party_size"] = data[["adults", "children", "babies"]].fillna(0).sum(axis=1).astype(int)
	data["arrival_month"] = data["arrival_date_month"].astype(str)
	month_number = data["arrival_month"].map(MONTH_NUMBER)
	data["arrival_date"] = (
		data["arrival_date_year"].astype(int).astype(str)
		+ "-" + month_number
		+ "-" + data["arrival_date_day_of_month"].astype(int).astype(str).str.zfill(2)
	)
	converter = coco.CountryConverter()
	country_codes = data["country"].dropna().unique().tolist()
	country_aliases = {"TMP": "TLS"}
	continent_values = converter.convert(
		names=[country_aliases.get(country, country) for country in country_codes],
		src="ISO3",
		to="continent",
		not_found="Unknown",
	)
	continent_by_country = dict(zip(country_codes, continent_values))
	data["continent"] = data["country"].map(continent_by_country).fillna("Unknown")
	data["party_size_bucket"] = data["party_size"].map(_party_size_bucket)

	scenarios = [
		_k_statistics(
			data,
			"exact date, country, party size",
			["hotel", "country_qi", "arrival_date", "party_size"],
		),
		_k_statistics(
			data,
			"month only, country, exact party size",
			["hotel", "country_qi", "arrival_month", "party_size"],
		),
		_k_statistics(
			data,
			"month, continent, generalized party size",
			["hotel", "continent", "arrival_month", "party_size_bucket"],
		),
	]
	results = pd.DataFrame(scenarios)
	results.to_csv(output / "privacy_k_anonymity.csv", index=False)
	(output / "privacy_summary.json").write_text(
		json.dumps(
			{
				"results": scenarios,
				"interpretation_limit": (
					"These are record-level uniqueness estimates for selected quasi-identifiers, "
					"not proof of individual re-identification. k-anonymity does not prevent "
					"attribute disclosure or replace access controls and lawful processing."
				),
			},
			indent=2,
		)
	)

	fig, axis = plt.subplots(figsize=(9, 5))
	positions = range(len(results))
	axis.bar(positions, results["share_k_1"], label="k = 1", color="#d1495b")
	axis.bar(positions, results["share_k_below_5"], label="k < 5", color="#edae49", alpha=0.8)
	axis.set_xticks(list(positions), results["scenario"], rotation=20, ha="right")
	axis.set(ylabel="Share of records", title="Uniqueness under quasi-identifier generalization")
	axis.set_ylim(0, 1)
	axis.legend()
	fig.tight_layout()
	fig.savefig(figures / "privacy_k_anonymity.png", dpi=160)
	plt.close(fig)
	print(results.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
	print(f"Privacy outputs saved to {output}")
	return results


if __name__ == "__main__":
	run_privacy_analysis()