import matplotlib.pyplot as plt

def plot_results(X, y, regressor, title, xlabel, ylabel):
    plt.scatter(X, y, color='red')
    plt.plot(X, regressor.predict(X), color='blue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
