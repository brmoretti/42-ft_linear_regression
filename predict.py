import pandas as pd
from LinearRegression import LinearRegression


def load_thetas() -> tuple[float, float]:
    try:
        table = pd.read_csv("./thetas.csv").astype(float)
        return table.iloc[-1, 0], table.iloc[-1, 1]
    except Exception as e:
        print(f"{e} - using theta0 = 0.0 and theta1 = 0.0")
        return 0.0, 0.0


def main():
    theta0, theta1 = 0.0, 0.0
    try:
        theta0, theta1 = load_thetas()
        mileage = float(input("Enter the mileage: "))
        assert mileage > 0, "mileage should be greater than zero"
        model = LinearRegression(theta0, theta1)
        print(f"Predicted price $ {model.predict(mileage):.2f}")
    except AssertionError as e:
        print(f"{e}")
    except Exception as e:
        print(f"{e}")


if __name__ == "__main__":
    main()
