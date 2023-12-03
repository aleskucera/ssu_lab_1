import numpy as np


class LSCPairWiseDependency:
    def __init__(self, X_train: list, y_train: list, X_test: list, y_test: list, mapping: dict):
        # Dataset
        self.X_train, self.y_train = None, None
        self.X_test, self.y_test = None, None

        self.mapping = mapping
        self.create_dataset(X_train, y_train, X_test, y_test)

        # Model
        self.W = np.zeros([self.num_features, self.num_letters], dtype=np.float32)
        self.G = np.zeros([self.num_letters, self.num_letters], dtype=np.float32)

    @property
    def num_letters(self):
        return len(self.mapping['characters'])

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
        indices = [int(self.mapping['characters'][c]) for c in name]
        return np.array(indices, dtype=np.uint8)

    def create_dataset(self, X_train: list, y_train: list, X_test: list, y_test: list) -> None:
        self.X_train = []
        for i in range(len(X_train)):
            self.X_train.append(
                np.concatenate([X_train[i].transpose(), np.ones((X_train[i].transpose().shape[0], 1))], axis=1))

        self.X_test = []
        for i in range(len(X_test)):
            self.X_test.append(
                np.concatenate([X_test[i].transpose(), np.ones((X_test[i].transpose().shape[0], 1))], axis=1))

        self.y_train = []
        for name in y_train:
            n = self.mapping['names'][name]
            letters = self.letters(n)
            self.y_train.append(letters)

        self.y_test = []
        for name in y_test:
            letters = self.letters(self.mapping['names'][name])
            self.y_test.append(letters)

    def train(self):
        counter = 0
        while True:
            end = True
            for i in range(self.num_train_samples):
                sample, label = self.X_train[i], self.y_train[i]
                prediction = self.predict(self.X_train[i])
                if not np.array_equal(label, prediction):
                    self.update(sample, label, prediction)
                    end = False
            counter += 1
            if counter % 10 == 0:
                print(f"Train accuracy: {self.evaluate_train()}")
            if end:
                break
        print(f"Training completed with train accuracy: {self.evaluate_train()}")


    def predict(self, X: np.ndarray) -> np.ndarray:
        name_length = len(X)
        y_pred = np.zeros(name_length, dtype=np.uint8)

        Q = np.dot(X, self.W)  # (name_length, num_letters)
        F = np.zeros_like(Q)

        # In first iteration assign to each node the maximum
        # cost of the path from the start node to it (F)
        F[0] = Q[0]
        for i in range(1, name_length):
            F[i] = Q[i] + np.max(F[i - 1][:, np.newaxis] + self.G, axis=0)

        # Now search with the greedy algorithm the path with the
        # maximum cost from the end node to the start node
        y_pred[-1] = np.argmax(F[-1])
        for i in range(name_length - 2, -1, -1):
            y_pred[i] = np.argmax(F[i] + self.G[:, y_pred[i + 1]])

        return y_pred


    def predict_old(self, X: np.ndarray) -> np.ndarray:
        y_pred = np.zeros(len(X), dtype=np.uint8)

        scores = np.dot(X, self.W)
        for i in range(1, len(X)):
            y_pred[i-1] = np.argmax(scores[i-1])
            scores[i] += self.g[y_pred[i-1]]

        y_pred[-1] = np.argmax(scores[-1])
        return y_pred.astype(np.uint8)


    def update(self, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        for i in range(len(X)):
            self.W[:, int(y_true[i])] += X[i]
            self.W[:, int(y_pred[i])] -= X[i]

            if i > 0:
                self.G[int(y_true[i-1]), int(y_true[i])] += 1
                self.G[int(y_pred[i-1]), int(y_pred[i])] -= 1

    def evaluate_test(self) -> float:
        correct = np.zeros([self.num_test_samples])
        for i in range(self.num_test_samples):
            prediction = self.predict(self.X_test[i])
            correct[i] = np.array_equal(prediction, self.y_test[i])
        accuracy = np.sum(correct) / self.num_test_samples
        return accuracy

    def evaluate_train(self) -> float:
        correct = np.zeros([self.num_train_samples])
        for i in range(self.num_train_samples):
            prediction = self.predict(self.X_train[i])
            correct[i] = np.array_equal(prediction, self.y_train[i])
        accuracy = np.sum(correct) / self.num_train_samples
        return accuracy