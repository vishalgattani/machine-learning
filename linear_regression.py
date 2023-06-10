import numpy as np
import matplotlib.pyplot as plt
import math

class MyLinearRegression():
    def __init__(self,X_train,y_train) -> None:
        self.X = X_train
        self.y = y_train
        self.num_features = self.X.shape[1]
        self.theta = np.zeros([1, self.num_features])
        self.cost = 0
        self.final_theta = np.zeros([1, self.num_features])

    def fit(self,X_train,y_train):
        self.final_theta, self.cost = gradientdescent(X_train, y_train,self.theta)

    def predict(self,X_test):
        return X_test @ self.final_theta.T



# Define computecost function
def computecost(X, y, theta):
    H = X @ theta.T
    J = np.power((H - y), 2)
    sum = np.sum(J)/(2 * len(X))
    return sum

# Define gradientdescent function
def gradientdescent(X, y, theta, iterations=1000, alpha=0.01):
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