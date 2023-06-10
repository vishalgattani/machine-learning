import math

def error_metrics(y_pred,y_test):
    mae_sum = 0
    mse_sum = 0
    mape_sum = 0
    mpe_sum = 0
    for i in range(len(y_pred)):
        mae_sum += abs(y_test[i] - y_pred[i])
        mse_sum += (y_test[i] - y_pred[i])**2
        mape_sum += (abs((y_test[i] - y_pred[i]))/y_test[i])
        mpe_sum += ((y_test[i] - y_pred[i])/y_test[i])
    mae = mae_sum / len(y_pred)
    mse = mse_sum / len(y_pred)
    mape = mape_sum / len(y_pred)
    mpe = mpe_sum / len(y_pred)

    error_metrics = {"mae":mae[0],
                    "mse":mse[0],
                    "rmse":math.sqrt(mse[0]),
                    "mape":mape[0],
                    "mpe":mpe[0]}

    return error_metrics
