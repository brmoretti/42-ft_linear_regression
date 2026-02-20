import pandas as pd
import numpy as np
import math
from pathlib import Path


class LinearRegression:
    def __init__(self, theta0: float = 0.0, theta1: float = 0.0):
        self._theta0 = float(theta0)
        self._theta1 = float(theta1)
        self._learning_rate = 0.1
        self._data = None
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

        self._data = table.astype(float)
        self._row_count = self._data.shape[0]

    def train(self, n_iterations: int = 1000) -> None:
        if not isinstance(n_iterations, int):
            raise TypeError("n_iterations should be an integer")
        if n_iterations <= 0:
            raise ValueError("n_iterations should be greater than 0")

        if self._data is None or self._row_count == 0:
            raise AttributeError(
                "Training data not loaded. Call load_data() before train()."
            )

        first_term = self._learning_rate / self._row_count
        col_x = self._data.iloc[:, 0].to_numpy(dtype=np.float64)
        col_y = self._data.iloc[:, 1].to_numpy(dtype=np.float64)

        for _ in range(n_iterations):
            print(f"{self._theta0}      {self._theta1}")
            predictions = self._theta0 + self._theta1 * col_x
            error = predictions - col_y
            self._theta0 = first_term * sum(error)
            self._theta1 = first_term * sum(error * col_x)

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
