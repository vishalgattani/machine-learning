import numpy as np
import matplotlib.pyplot as plt
import math

class MyLinearRegression():
    def __init__(self,X_train,y_train) -> None:
        """initilializa class with num features and training data

        Args:
            X_train (df): features to train
            y_train (np.array): targets to train on
        """
        self.X = X_train
        self.y = y_train
        self.num_features = self.X.shape[1]
        self.theta = np.zeros([1, self.num_features])
        self.cost = 0
        self.final_theta = np.zeros([1, self.num_features])

    def fit(self,X_train,y_train):
        """compute gradient descent

        Args:
            X_train (_type_): _description_
            y_train (_type_): _description_
        """
        self.final_theta, self.cost = gradientdescent(X_train, y_train,self.theta)

    def predict(self,X_test):
        return X_test @ self.final_theta.T



# Define computecost function
def computecost(X, y, theta):
    """cost function - desirable to be low in value
    Mean Squared error and not mean absolute error

    Args:
        X (df): training features
        y (np.array): target feature
        theta (_type_): feature vector coeffs

    Returns:
        float: cost value
    """
    #
    H = X @ theta.T
    J = np.power((H - y), 2)
    sum = np.sum(J)/(2 * len(X))
    return sum

# Define gradientdescent function
def gradientdescent(X, y, theta, iterations=1000, alpha=0.01):
    """_summary_

    Args:
        X (df): features to train
        y (np.array): target to train on
        theta (_type_): initial feature vector coeffs
        iterations (int, optional): num iterations. Defaults to 1000.
        alpha (float, optional): learning rate. Defaults to 0.01.

    Returns:
        _type_: feature vector coeffs and cost at each iteration
    """
    cost = np.zeros(iterations)
    for i in range(iterations):
        H = X @ theta.T
        theta = theta - (alpha/len(X)) * np.sum(X * (H - y), axis=0)
        cost[i] = computecost(X, y, theta)

    # Plot Iterations vs. Cost figure
    fig_2, ax = plt.subplots()
    ax.plot(np.arange(iterations), cost, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Iterations vs. Cost')
    plt.show()
    return theta, cost

def metrics(predictions, Y_test):
    """_summary_

    Args:
        predictions (np.array): predicted values
        Y_test (np.array): target values

    Returns:
        float: mae, mse, rmse, and
        rsquare error:- R-squared is a statistical measure that indicates how much of the variation of a dependent variable is explained by an independent variable in a regression model.
    """
    #calculating mean absolute error
    MAE = np.mean(np.abs(predictions-Y_test))

    #calculating root mean square error
    MSE = np.square(np.subtract(Y_test,predictions)).mean()
    RMSE = math.sqrt(MSE)

    #calculating r_square
    rss = np.sum(np.square((Y_test- predictions)))
    mean = np.mean(Y_test)
    sst = np.sum(np.square(Y_test-mean))
    r_square = 1 - (rss/sst)


    return MAE, RMSE, r_square