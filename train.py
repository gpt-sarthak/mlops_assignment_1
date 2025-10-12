from sklearn.tree import DecisionTreeRegressor
from misc import load_data, preprocess_data, train_model, evaluate_model

def main():
    # Load and preprocess data
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Initialize, train, and evaluate the Decision Tree model
    dt_model = DecisionTreeRegressor(random_state=42)
    dt_model = train_model(dt_model, X_train, y_train)
    mse = evaluate_model(dt_model, X_test, y_test)
    print(1)
    print(f"Decision Tree Regressor - Average MSE: {mse:.2f}")

if __name__ == "__main__":
    main()