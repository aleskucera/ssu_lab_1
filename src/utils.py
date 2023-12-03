from os import listdir
from os.path import isfile, join

import yaml
import numpy as np
from PIL import Image


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