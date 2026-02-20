from LinearRegression import LinearRegression


def main():
    model = LinearRegression()
    model.load_data("./data.csv")
    model.train(10)
    model.save_thetas()


if __name__ == "__main__":
    main()
