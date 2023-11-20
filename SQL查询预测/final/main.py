from data import get_train_data, get_test_data
from data import write_data
from train import sk_train
from train import network_train
import numpy as np


if __name__ == "__main__":
    train_x, train_y, scaler = get_train_data("training_data.csv", False)
    test_x = get_test_data("testing_data.csv", scaler)
    model_num = 5
    result1 = np.array(sk_train(model_num, train_x, train_y, test_x))
    #result2 = np.array(network_train("training_data.csv", "testing_data.csv", 100000, 100, 1024, 256, True))
    write_data(result1, "result/testing_result.csv")