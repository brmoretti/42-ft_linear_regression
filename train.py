from LinearRegression import LinearRegression


def main():
    model = LinearRegression()
    model.learning_rate = 0.1
    model.load_data("./data.csv")
    model.train(5000)
    model.save_thetas()


if __name__ == "__main__":
    main()
