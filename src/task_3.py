import numpy as np


class LSCForFixedNumberOfSequences:
    def __init__(self, X_train: list, y_train: list, X_test: list, y_test: list, mapping: dict):

        # Dataset
        self.X_train, self.y_train = None, None
        self.X_test, self.y_test = None, None

        self.mapping = mapping
        self.create_dataset(X_train, y_train, X_test, y_test)

        # Model
        self.W = np.zeros([self.num_features, self.num_letters], dtype=np.float32)
        self.v = np.zeros(self.num_classes, dtype=np.float32)

    @property
    def num_letters(self):
        return len(self.mapping['characters'])

    @property
    def num_classes(self):
        return len(self.mapping['names'])

    @property
    def num_features(self):
        return self.X_train[0].shape[1]

    @property
    def num_train_samples(self):
        return len(self.X_train)

    @property
    def num_test_samples(self):
        return len(self.X_test)

    def letters(self, y: int):
        """Return the indices of the letters of a given name in the dataset.
        """
        name = self.mapping['names_inverse'][y]
        indices = [self.mapping['characters'][c] for c in name]
        return indices

    def create_dataset(self, X_train: list, y_train: list, X_test: list, y_test: list) -> None:
        self.X_train = []
        for i in range(len(X_train)):
            self.X_train.append(
                np.concatenate([X_train[i].transpose(), np.ones((X_train[i].transpose().shape[0], 1))], axis=1))
        self.X_test = []
        for i in range(len(X_test)):
            self.X_test.append(
                np.concatenate([X_test[i].transpose(), np.ones((X_test[i].transpose().shape[0], 1))], axis=1))

        # map labels to integers
        self.y_train = np.zeros([len(y_train)], dtype=np.uint8)
        for i in range(len(y_train)):
            self.y_train[i] = int(self.mapping['names'][y_train[i]])

        self.y_test = np.zeros([len(y_test)], dtype=np.uint8)
        for i in range(len(y_test)):
            self.y_test[i] = int(self.mapping['names'][y_test[i]])

    def train(self):
        counter = 0
        while True:
            end = True
            for i in range(self.num_train_samples):
                sample, y_true = self.X_train[i], self.y_train[i]
                y_pred = self.predict(self.X_train[i])
                if y_pred != y_true:
                    self.update(sample, y_true, y_pred)
                    end = False
            counter += 1
            if counter % 10 == 0:
                print(f"Train accuracy: {self.evaluate_train()}")
            if end:
                break
        print(f"Training completed with train accuracy: {self.evaluate_train()}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = np.zeros([self.num_classes])
        possible_classes = [c for c in range(self.num_classes) if len(self.letters(c)) == X.shape[0]]
        for c in possible_classes:
            letters = self.letters(c)
            for i, l in enumerate(letters):
                scores[c] += np.dot(X[i], self.W[:, l])
            scores[c] += self.v[c]

        y_pred = np.argmax(scores)
        if len(self.letters(y_pred)) != X.shape[0]:
            y_pred = np.random.choice(possible_classes)
        return y_pred

    def update(self, X: np.ndarray, y_true: int, y_pred: int) -> None:
        letters_true = self.letters(y_true)
        letters_pred = self.letters(y_pred)
        for i, l in enumerate(letters_true):
            self.W[:, l] += X[i]
        for i, l in enumerate(letters_pred):
            self.W[:, l] -= X[i]

        self.v[y_true] += 1
        self.v[y_pred] -= 1

    def evaluate_test(self):
        y_pred = np.zeros([self.num_test_samples])
        for i in range(self.num_test_samples):
            y_pred[i] = self.predict(self.X_test[i])
        accuracy = np.sum(y_pred == self.y_test) / self.num_test_samples
        return accuracy

    def evaluate_train(self):
        y_pred = np.zeros([self.num_train_samples])
        for i in range(self.num_train_samples):
            y_pred[i] = self.predict(self.X_train[i])
        accuracy = np.sum(y_pred == self.y_train) / self.num_train_samples
        return accuracy