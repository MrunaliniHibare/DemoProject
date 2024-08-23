from data_processing import load_data, split_data
from model import train_model, predict_model
from visualization import plot_results

def main():
    X, y = load_data('Salary_Data.csv')
    X_train, X_test, y_train, y_test = split_data(X, y)
    regressor = train_model(X_train, y_train)
    y_pred = predict_model(regressor, X_test)
    
    plot_results(X_train, y_train, regressor, 'Salary vs Experience (Training set)', 'Years of Experience', 'Salary')
    plot_results(X_test, y_test, regressor, 'Salary vs Experience (Test set)', 'Years of Experience', 'Salary')

if __name__ == "__main__":
    main()
