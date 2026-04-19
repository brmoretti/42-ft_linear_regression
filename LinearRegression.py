import pandas as pd
import numpy as np
import math
from pathlib import Path
from typing import Optional


class LinearRegression:
    def __init__(self, theta0: float = 0.0, theta1: float = 0.0):
        self._theta0 = float(theta0)
        self._theta1 = float(theta1)
        self._learning_rate = 0.1
        self._data: Optional[pd.DataFrame] = None
        self._row_count = 0

    @property
    def coefficients(self) -> tuple[float, float]:
        return self._theta0, self._theta1

    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate: float) -> None:
        if not isinstance(learning_rate, (int, float)):
            raise TypeError("learning_rate should be a number")

        if not math.isfinite(float(learning_rate)) or learning_rate <= 0:
            raise ValueError("learning_rate should be greater than 0")

        self._learning_rate = float(learning_rate)

    def predict(self, x: float) -> float:
        value = float(x)
        return self._theta0 + self._theta1 * value

    def load_data(self, csv_file_path: str) -> None:
        table = pd.read_csv(csv_file_path, header=0)

        if table.shape[0] == 0:
            raise ValueError("CSV file is empty")

        if table.shape[1] != 2:
            raise ValueError(f"Expected 2 columns, got {table.shape[1]}")

        table = table.copy()
        for column in table.columns:
            table[column] = pd.to_numeric(table[column], errors="coerce")

        if table.isnull().any().any():
            raise ValueError("Data contains missing or non-numeric values")

        table_float = table.astype(float)
        self._data = table_float
        self._row_count = table_float.shape[0]

    def _validate_training_request(self, n_iterations: int) -> None:
        if not isinstance(n_iterations, int):
            raise TypeError("n_iterations should be an integer")
        if n_iterations <= 0:
            raise ValueError("n_iterations should be greater than 0")
        if self._data is None or self._row_count == 0:
            raise AttributeError(
                "Training data not loaded. Call load_data() before train()."
            )

    def _extract_training_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if self._data is None:
            raise AttributeError(
                "Training data not loaded. Call load_data() before train()."
            )

        mileage_raw = self._data.iloc[:, 0].to_numpy(dtype=np.float64)
        price = self._data.iloc[:, 1].to_numpy(dtype=np.float64)
        return mileage_raw, price

    def _normalize_mileage(
        self, mileage_raw: np.ndarray
    ) -> tuple[np.ndarray, float, float]:
        min_mileage = float(np.min(mileage_raw))
        max_mileage = float(np.max(mileage_raw))
        mileage_range = max_mileage - min_mileage
        if mileage_range == 0:
            raise ValueError("All mileage values are identical; cannot train")
        mileage = (mileage_raw - min_mileage) / mileage_range
        return mileage, min_mileage, mileage_range

    def _to_normalized_thetas(
        self, min_mileage: float, mileage_range: float
    ) -> tuple[float, float]:
        theta0_n = self._theta0 + self._theta1 * min_mileage
        theta1_n = self._theta1 * mileage_range
        return theta0_n, theta1_n

    def _gradient_step(
        self,
        theta0_n: float,
        theta1_n: float,
        mileage: np.ndarray,
        price: np.ndarray,
        alpha_over_m: float,
    ) -> tuple[float, float]:
        estimated_price = theta0_n + theta1_n * mileage
        errors = estimated_price - price

        tmp_theta0 = alpha_over_m * np.sum(errors)
        tmp_theta1 = alpha_over_m * np.sum(errors * mileage)

        new_theta0_n = theta0_n - tmp_theta0
        new_theta1_n = theta1_n - tmp_theta1
        return new_theta0_n, new_theta1_n

    def _assert_finite_thetas(self, theta0_n: float, theta1_n: float) -> None:
        if not np.isfinite(theta0_n) or not np.isfinite(theta1_n):
            raise FloatingPointError(
                "Numerical overflow during training; reduce learning_rate"
            )

    def _from_normalized_thetas(
        self,
        theta0_n: float,
        theta1_n: float,
        min_mileage: float,
        mileage_range: float,
    ) -> tuple[float, float]:
        theta1 = theta1_n / mileage_range
        theta0 = theta0_n - (theta1_n * min_mileage / mileage_range)
        return theta0, theta1

    def train(self, n_iterations: int = 1000) -> None:
        self._validate_training_request(n_iterations)

        alpha_over_m = self._learning_rate / self._row_count
        mileage_raw, price = self._extract_training_arrays()
        mileage, min_mileage, mileage_range = self._normalize_mileage(
            mileage_raw
        )
        theta0_n, theta1_n = self._to_normalized_thetas(
            min_mileage, mileage_range
        )

        for _ in range(n_iterations):
            theta0_n, theta1_n = self._gradient_step(
                theta0_n, theta1_n, mileage, price, alpha_over_m
            )
            self._assert_finite_thetas(theta0_n, theta1_n)

        self._theta0, self._theta1 = self._from_normalized_thetas(
            theta0_n, theta1_n, min_mileage, mileage_range
        )

    def save_thetas(self) -> None:
        file_path = Path("./thetas.csv")
        row = pd.DataFrame(
            [{"theta0": self._theta0, "theta1": self._theta1}],
            columns=["theta0", "theta1"],
        )

        row.to_csv(
            file_path,
            mode="a",
            header=not file_path.exists() or file_path.stat().st_size == 0,
            index=False,
        )
