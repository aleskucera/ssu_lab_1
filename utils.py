from os import listdir
from os.path import isfile, join

import yaml
import numpy as np
from PIL import Image


def create_dataset_task_1(X_train: list, y_train: list, X_test: list, y_test: list, mapping: dict) -> tuple:
    """Create a dataset for character recognition.

    Args:
        X_train: list of length M_train of numpy arrays of shape (D, L_i) where D is the number of features
            and L_i is the number of letters in the i-th example.
        y_train: list of length M_test of labels (strings)
        X_test: list of length M_test of numpy arrays of shape (D, L_i) where D is the number of features
            and L_i is the number of letters in the i-th example.
        y_test: list of length M_test of labels (strings)
        mapping: dictionary with all four mappings

    Returns:
        X_train: numpy array of shape (N_train, D + 1) where N_train is M_train * sum(L_i)
        y_train: numpy array of shape (N_train)
        X_test: numpy array of shape (N_test, D + 1) where N_test is M_test * sum(L_i)
        y_test: numpy array of shape (N_test)
    """

    X_train_concat = np.concatenate(X_train, axis=1).transpose()
    X_test_concat = np.concatenate(X_test, axis=1).transpose()

    # Add ones for bias
    X_train_concat = np.concatenate([X_train_concat, np.ones((X_train_concat.shape[0], 1))], axis=1)
    X_test_concat = np.concatenate([X_test_concat, np.ones((X_test_concat.shape[0], 1))], axis=1)

    # map characters to integers
    y_train_ = np.array([], dtype=np.uint8)
    for i in range(len(y_train)):
        chars = list(y_train[i])
        chars_ = np.array([mapping['characters'][chars[j]] for j in range(len(chars))], dtype=np.uint8)
        y_train_ = np.concatenate([y_train_, chars_])

    y_test_ = np.array([], dtype=np.uint8)
    for i in range(len(y_test)):
        chars = list(y_test[i])
        chars_ = np.array([mapping['characters'][chars[j]] for j in range(len(chars))], dtype=np.uint8)
        y_test_ = np.concatenate([y_test_, chars_])

    return X_train_concat, y_train_, X_test_concat, y_test_


def create_dataset_task_2(X_train: list, y_train: list, X_test: list, y_test: list) -> tuple:
    """Create a dataset for handwritten names recognition.

    Args:
        X_train: list of length N_train of numpy arrays of shape (D, L_i) where D is the number of features
            and L_i is the number of letters in the i-th example.
        y_train: list of length N_test of labels (strings)
        X_test: list of length N_test of numpy arrays of shape (D, L_i) where D is the number of features
            and L_i is the number of letters in the i-th example.
        y_test: list of length N_test of labels (strings)
        mapping: dictionary with all four mappings

    Returns:
        X_train: list of length N_train of numpy arrays of shape (L_i, D + 1) where D is the number of features
            and L_i is the number of letters in the i-th example.
        y_train: numpy array of shape (N_train)
        X_test: list of length N_test of numpy arrays of shape (L_i, D + 1) where D is the number of features
            and L_i is the number of letters in the i-th example.
        y_test: numpy array of shape (N_test)
    """

    X_train_ = []
    for i in range(len(X_train)):
        X_train_.append(np.concatenate([X_train[i].transpose(), np.ones((X_train[i].transpose().shape[0], 1))], axis=1))
    X_test_ = []
    for i in range(len(X_test)):
        X_test_.append(np.concatenate([X_test[i].transpose(), np.ones((X_test[i].transpose().shape[0], 1))], axis=1))

    return X_train_, y_train, X_test_, y_test

def create_dataset_task_3(X_train: list, y_train: list, X_test: list, y_test: list, mapping: dict) -> tuple:
    X_train_ = []
    for i in range(len(X_train)):
        X_train_.append(np.concatenate([X_train[i].transpose(), np.ones((X_train[i].transpose().shape[0], 1))], axis=1))
    X_test_ = []
    for i in range(len(X_test)):
        X_test_.append(np.concatenate([X_test[i].transpose(), np.ones((X_test[i].transpose().shape[0], 1))], axis=1))

    # map labels to integers
    y_train_ = np.zeros([len(y_train)])
    for i in range(len(y_train)):
        y_train_[i] = mapping['names'][y_train[i]]

    y_test_ = np.zeros([len(y_test)])
    for i in range(len(y_test)):
        y_test_[i] = mapping['names'][y_test[i]]

    return X_train_, y_train_, X_test_, y_test_





def perceptron_algorithm_1(X: np.ndarray, y: np.ndarray, num_classes: int) -> np.ndarray:
    """Perceptron algorithm for classification of handwritten digits. The data are
    linearly separable and the algorithm is guaranteed to converge.

    The update rule is
        w <- w + x_i    if y_i * w^T x_i <= 0

    Args:
        X: numpy array of shape (N_train, D) where N_train is the number of training examples
            and D is the number of features
        y: numpy array of shape (N_train) with labels

    Returns:
        w: numpy array of shape (D) with the final weights
    """

    num_samples = X.shape[0]
    num_features = X.shape[1]

    W = np.zeros([num_features, num_classes])
    while True:
        end = True
        for i in range(num_samples):
            scores = np.dot(X[i], W)
            y_pred = np.argmax(scores)
            y_true = y[i]
            if y_pred != y_true:
                W[:, y_true] += X[i]
                W[:, y_pred] -= X[i]
                end = False
        if end:
            break
    return W


def evaluate_task_1(X, y, W):
    num_samples = X.shape[0]

    y_pred = np.zeros([num_samples])
    for i in range(num_samples):
        scores = np.dot(X[i], W)
        y_pred[i] = np.argmax(scores)

    accuracy = np.sum(y_pred == y) / num_samples
    return accuracy


class DependencyScore:
    def __init__(self, X_train, y_train, mapping, num_characters=26):
        self.X_train = X_train
        self.y_train = y_train
        self.mapping = mapping
        self.num_characters = num_characters
        self.dependency_matrix = self.compute_dependency_matrix()

    def compute_dependency_matrix(self):
        """Compute the probability of two letters being next to each other.
        The [i, j] element of the matrix is the probability of the j-th letter
        being after the i-th letter.
        """

        dependency_matrix = np.zeros([self.num_characters, self.num_characters])
        for name in self.y_train:
            chars = list(name)
            for j in range(len(chars) - 1):
                first_letter = self.mapping['characters'][chars[j]]
                second_letter = self.mapping['characters'][chars[j+1]]
                dependency_matrix[first_letter, second_letter] += 1

        s = np.sum(dependency_matrix, axis=1, keepdims=True)
        for i in range(len(s)):
            if s[i] != 0:
                dependency_matrix[i] /= s[i]
        return dependency_matrix

    def __call__(self, letter_index):
        return self.dependency_matrix[int(letter_index)]

def linearly_pairwise_classify(image: np.ndarray, W: np.ndarray, g: DependencyScore, mapping: dict) -> str:
    y_pred = np.zeros(len(image))
    scores = np.zeros([len(image), W.shape[1]])
    F = np.zeros([len(image)])

    # Compute the first sample
    scores[0] = np.dot(image[0], W)
    y_pred[0] = np.argmax(scores[0])
    F[0] = np.max(scores[0])

    # Compute the rest of the name
    for i in range(1, len(image)):
        scores[i] = F[i-1] + g(y_pred[i-1])
        y_pred[i] = np.argmax(scores[i])
        F[i] = np.max(scores[i]) + np.dot(image[i], W[:, int(y_pred[i])])

    name_pred = ''
    for num in y_pred:
        name_pred += mapping['characters_inverse'][num]
    return name_pred

def perceptron_algorithm_2(X: np.ndarray, y: np.ndarray, num_classes: int, mapping: dict) -> np.ndarray:

    """Perceptron algorithm for classification of handwritten digits. The data are
    linearly separable and the algorithm is guaranteed to converge.

    The update rule is
        w <- w + x_i    if y_i * w^T x_i <= 0

    Args:
        X: numpy array of shape (N_train, D) where N_train is the number of training examples
            and D is the number of features
        y: numpy array of shape (N_train) with labels

    Returns:
        w: numpy array of shape (D) with the final weights
    """

    num_samples = len(X)
    num_features = X[0].shape[1]

    g = DependencyScore(X, y, mapping, num_characters=num_classes)

    W = np.zeros([num_features, num_classes])
    counter = 0
    while True:
        end = True
        for i in range(num_samples):
            y_pred = linearly_pairwise_classify(X[i], W, g, mapping)
            y_true = y[i]
            for j, (char_true, char_pred) in enumerate(zip(y_true, y_pred)):
                if char_true != char_pred:
                    W[:, mapping['characters'][char_true]] += X[i][j]
                    W[:, mapping['characters'][char_pred]] -= X[i][j]
                    end = False
                    break
        if end:
            break
        counter += 1
        print(counter)

        # every 10 iterations, evaluate the accuracy
        if counter % 10 == 0:
            print(f"Accuracy: {evaluate_task_2(X, y, W, mapping)}")
    return W

def evaluate_task_2(X_test, y_test, W, mapping):
    num_samples = len(X_test)
    g = DependencyScore(X_test, y_test, mapping)

    score = {"right": 0, "wrong": 0}
    for i in range(num_samples):
        y_pred = linearly_pairwise_classify(X_test[i], W, g, mapping)
        y_true = y_test[i]
        if y_pred == y_true:
            score["right"] += 1
        else:
            score["wrong"] += 1

    accuracy = score["right"] / (score["right"] + score["wrong"])

    return accuracy

def perceptron_algorithm_3(X: np.ndarray, y: np.ndarray, num_classes: int, mapping: dict) -> np.ndarray:

    num_samples = len(X)
    num_features = X[0].shape[1]

    W = np.zeros([num_features, num_classes])

    name_scores = np.zeros([len(mapping['names'])])
    for i in range(num_samples):
        name_scores[int(y[i])] += 1

    while True:
        end = True
        for i in range(num_samples):
            scores = np.dot(X[i], W)
            y_pred = np.argmax(scores)
            y_true = y[i]
            if y_pred != y_true:
                W[:, int(y_true)] += X[i]
                W[:, y_pred] -= X[i]
                end = False
        if end:
            break







