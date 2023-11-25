from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import isfile, join
import yaml

def create_mapping(Y: list) -> dict:
    """Create a mapping from the labels to integers.

    Mapping:
    1. Alphabet to integers (characters)
    2. Integers to alphabet (characters_inverse)
    3. Names to integers (labels)
    4. Integers to names (labels_inverse)

    Args:
        Y: list of labels (strings)

    Returns:
        mapping: dictionary with all four mappings
    """

    mapping = {}

    # alphabet to integers
    alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
                'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                'u', 'v', 'w', 'x', 'y', 'z']

    mapping['characters'] = {alphabet[i]: i for i in range(len(alphabet))}

    # integers to alphabet
    mapping['characters_inverse'] = {i: alphabet[i] for i in range(len(alphabet))}

    # names to integers
    labels = list(set(Y))
    labels.sort()
    mapping['names'] = {labels[i]: i for i in range(len(labels))}

    # integers to names
    mapping['names_inverse'] = {i: labels[i] for i in range(len(labels))}

    with open('mapping.yaml', 'w') as file:
        yaml.dump(mapping, file)

    return mapping


def create_character_dataset(X_train: list, y_train: list, X_test: list, y_test: list, mapping: dict) -> tuple:
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
        X_train: numpy array of shape (N_train, D) where N_train is M_train * sum(L_i)
        y_train: numpy array of shape (N_train)
        X_test: numpy array of shape (N_test, D) where N_test is M_test * sum(L_i)
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


def create_dataset(X_train: list, y_train: list, X_test: list, y_test: list, mapping: dict) -> tuple:
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
        X_train: list of length N_train of numpy arrays of shape (D, L_i) where D is the number of features
            and L_i is the number of letters in the i-th example.
        y_train: numpy array of shape (N_train)
        X_test: list of length N_test of numpy arrays of shape (D, L_i) where D is the number of features
            and L_i is the number of letters in the i-th example.
        y_test: numpy array of shape (N_test)
    """

    # map labels to integers
    y_train_ = np.zeros([len(y_train)])
    for i in range(len(y_train)):
        y_train_[i] = mapping['names'][y_train[i]]

    y_test_ = np.zeros([len(y_test)])
    for i in range(len(y_test)):
        y_test_[i] = mapping['names'][y_test[i]]

    return X_train, y_train_, X_test, y_test_



# load single example
def load_example(img_path):
    Y = img_path[img_path.rfind('_') + 1:-4]

    img = Image.open(img_path)
    img_mat = np.asarray(img)

    n_letters = len(Y)
    im_height = int(img_mat.shape[0])
    im_width = int(img_mat.shape[1] / n_letters)
    n_pixels = im_height * im_width

    X = np.zeros([int(n_pixels + n_pixels * (n_pixels - 1) / 2), n_letters])
    for i in range(n_letters):

        # single letter
        letter = img_mat[:, i * im_width:(i + 1) * im_width] / 255

        # compute features
        x = letter.flatten()
        X[0:len(x), i] = x
        cnt = n_pixels
        for j in range(0, n_pixels - 1):
            for k in range(j + 1, n_pixels):
                X[cnt, i] = x[j] * x[k]
                cnt = cnt + 1

        X[:, i] = X[:, i] / np.linalg.norm(X[:, i])

    return X, Y, img


# load all examples from a folder
def load_examples(image_folder):
    files = [f for f in listdir(image_folder) if isfile(join(image_folder, f))]

    X = []
    Y = []
    img = []
    for file in listdir(image_folder):
        path = join(image_folder, file)
        if isfile(path):
            X_, Y_, img_ = load_example(path)
            X.append(X_)
            Y.append(Y_)
            img.append(img_)

    return X, Y, img


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


def evaluate_task_1(X_test, y_test, W):
    num_samples = X_test.shape[0]
    num_features = X_test.shape[1]

    y_pred = np.zeros([num_samples])
    for i in range(num_samples):
        scores = np.dot(X_test[i], W)
        y_pred[i] = np.argmax(scores)

    accuracy = np.sum(y_pred == y_test) / num_samples
    return accuracy







