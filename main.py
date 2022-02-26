import csv
import math
import numpy as np

# Limits number of rows read from csv files
# Set LIMIT_ON to false to turn off row limit
LIMIT_ON = False
ROW_LIMIT = 1000

TRAINING_PATH = "train.csv"
TESTING_PATH = "test.csv"
HYPER_PATH = "hyper.csv"
OUT_PATH = "out.csv"
LOG_PATH = "log.csv"


def main():
    training_data = load_csv(TRAINING_PATH)
    training_features, training_labels = parse_csv_array(training_data)

    testing_data = load_csv(TESTING_PATH)
    testing_features, testing_labels = parse_csv_array(testing_data)

    hyperparameters = load_csv(HYPER_PATH)

    training_features, testing_features = preprocess_features(
        training_features, testing_features
    )

    # initialize weight vector to all zeros
    number_of_features = len(training_features[0])
    weight_vector = []
    for w in range(number_of_features):
        weight_vector.append(0)

    number_of_inputs = len(training_features)
    # train weight vector
    weight_vector = weight_train(
        weight_vector,
        hyperparameters,
        number_of_inputs,
        training_features,
        training_labels,
    )

    rmse = get_rmse(weight_vector, testing_features, testing_labels)

    csv_write(OUT_PATH, [[rmse]])

    return 0


def get_rmse(
    weight_vector: list, testing_features: list, testing_labels: list
) -> float:
    """Computes RMSE of testing data using weight vector trained on training data."""
    sum = 0
    for i in range(len(testing_features)):
        y_hat = get_y_hat(weight_vector, testing_features[i])
        sum += (testing_labels[i] - y_hat) ** 2

    return math.sqrt(sum / len(testing_labels))


def weight_train(
    weight_vector: list,
    hyperparameters: list,
    number_inputs: int,
    training_features: list,
    training_labels: list,
) -> list:
    """Optimizes weight vector using gradient descent."""
    alpha = float(hyperparameters[0][0])
    n1 = int(hyperparameters[1][0])
    n2 = int(hyperparameters[2][0])
    weight_log = []
    for iteration in range(n2):

        if iteration < n1:
            weight_log.append(weight_vector)
        new_weight_vector = []
        for w in range(len(weight_vector)):
            gradient = gradient_descent(
                weight_vector, training_features, training_labels, w
            )
            new_weight_vector.append(
                weight_vector[w] + (alpha / number_inputs) * gradient
            )
        weight_vector = new_weight_vector
    print_log(weight_log)
    return weight_vector


def print_log(log):
    for rows in log:
        for i in range(len(rows)):
            rows[i] = np.format_float_scientific(
                rows[i], precision=19, trim="k", min_digits=18, exp_digits=2
            )
    csv_write(LOG_PATH, log)
    return 0


def gradient_descent(
    weight_vector: list, training_features: list, training_labels: list, j: int
) -> float:
    """Calculates gradient for each weight"""
    sum = 0
    for i in range(len(training_features)):
        sum += (
            training_labels[i] - get_y_hat(weight_vector, training_features[i])
        ) * training_features[i][j]
    return sum


def get_y_hat(weight_vector: list, features: list) -> float:
    """Given a weight vector and set of features, computes y_hat"""
    y_hat = 0
    for i in range(len(features)):
        y_hat += weight_vector[i] * features[i]
    return y_hat


def load_csv(file_path: str) -> list:
    """Reads each row of csv file into an array"""
    with open(file_path) as file:
        csv_reader = csv.reader(file)
        c = 0
        out = []
        if ROW_LIMIT:
            for row in csv_reader:
                c += 1
                out.append(row)
                if c == ROW_LIMIT:
                    break
        else:
            for row in csv_reader:
                out.append(row)

    return out


def parse_csv_array(data_array: list) -> tuple:
    """Takes an array containing arrays of features and labels and splits into
    an array that contains an array of features for each input and a label for
    each input
    """
    features = []
    labels = []
    for row in data_array:
        row_length = len(row)
        features.append([float(row[feature]) for feature in range(0, row_length - 1)])
        labels.append(float(row[row_length - 1]))
    return features, labels


def csv_write(filename, rows):
    with open(filename, "w", newline="") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerows(rows)
    return 0


def preprocess_features(training_array: list, testing_array: list) -> tuple:
    """Handles normalization of features as well as adding an extra
    y-intercept feature."""
    training_array, testing_array = normalize_features(training_array, testing_array)
    for features in training_array:
        features.append(1)
    for features in testing_array:
        features.append(1)
    return training_array, testing_array


def normalize_features(training_array: list, testing_array: list) -> list:
    """Takes an array that contains arrays of features and
    normalizes each feature."""
    number_of_features = len(training_array[0])
    for i in range(number_of_features):
        feature_i = []
        for features in training_array:
            feature_i.append(features[i])
        min_val = min(feature_i)
        max_val = max(feature_i)
        for features in training_array:
            features[i] = (2 * ((features[i] - min_val) / (max_val - min_val))) - 1
        for features in testing_array:
            features[i] = (2 * ((features[i] - min_val) / (max_val - min_val))) - 1

    return training_array, testing_array


if __name__ == "__main__":
    main()
