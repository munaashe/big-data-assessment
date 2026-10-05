"""Build reproducible classification pipelines for cancellation prediction."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class QuantileClipper(BaseEstimator, TransformerMixin):
	"""Learn an upper winsorization bound from each training fold."""

	def __init__(self, quantile: float = 0.995) -> None:
		self.quantile = quantile

	def fit(self, values: np.ndarray, target: np.ndarray | None = None) -> "QuantileClipper":
		self.upper_bounds_ = np.quantile(values, self.quantile, axis=0)
		self.n_features_in_ = values.shape[1]
		return self

	def transform(self, values: np.ndarray) -> np.ndarray:
		return np.minimum(values, self.upper_bounds_)

	def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
		if input_features is None:
			input_features = [f"x{index}" for index in range(self.n_features_in_)]
		return np.asarray(input_features, dtype=object)


def build_models(
	numeric_features: list[str],
	categorical_features: list[str],
	random_state: int = 42,
) -> dict[str, Pipeline]:
	"""Create independent logistic-regression and random-forest pipelines."""
	def build_preprocessor() -> ColumnTransformer:
		other_numeric_features = [feature for feature in numeric_features if feature != "adr"]
		numeric_preprocessing = Pipeline(
			steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
		)
		adr_preprocessing = Pipeline(
			steps=[
				("imputer", SimpleImputer(strategy="median")),
				("clipper", QuantileClipper(quantile=0.995)),
				("scaler", StandardScaler()),
			]
		)
		categorical_preprocessing = Pipeline(
			steps=[
				("imputer", SimpleImputer(strategy="most_frequent")),
				("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20)),
			]
		)
		transformers = []
		if other_numeric_features:
			transformers.append(("numeric", numeric_preprocessing, other_numeric_features))
		if "adr" in numeric_features:
			transformers.append(("adr", adr_preprocessing, ["adr"]))
		if categorical_features:
			transformers.append(("categorical", categorical_preprocessing, categorical_features))
		return ColumnTransformer(
			transformers=transformers,
			remainder="drop",
		)

	return {
		"LogisticRegression": Pipeline(
			steps=[
				("preprocessing", build_preprocessor()),
				("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
			]
		),
		"RandomForest": Pipeline(
			steps=[
				("preprocessing", build_preprocessor()),
				(
					"classifier",
					RandomForestClassifier(
						n_estimators=150,
						min_samples_leaf=2,
						class_weight="balanced_subsample",
						n_jobs=-1,
						random_state=random_state,
					),
				),
			]
		),
	}
