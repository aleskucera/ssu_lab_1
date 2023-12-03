import numpy as np


class IndependentLinearClassifier:
    def __init__(self, X_train: list, y_train: list, X_test: list, y_test: list, mapping: dict):

        # Dataset
        self.X_train, self.y_train = None, None
        self.X_test, self.y_test = None, None

        self.mapping = mapping
        self.create_dataset(X_train, y_train, X_test, y_test)

        # Model
        self.W = np.zeros([self.num_features, self.num_classes], dtype=np.float32)


    @property
    def num_classes(self):
        return len(self.mapping['characters'])

    @property
    def num_features(self):
        return self.X_train.shape[1]

    @property
    def num_train_samples(self):
        return self.X_train.shape[0]

    @property
    def num_test_samples(self):
        return self.X_test.shape[0]


    def create_dataset(self, X_train: list, y_train: list, X_test: list, y_test: list) -> None:
        """Create a dataset for character recognition.

        Args:
            X_train: list of length M_train of numpy arrays of shape (D, L_i) where D is the number of features
                and L_i is the number of letters in the i-th example.
            y_train: list of length M_test of labels (strings)
            X_test: list of length M_test of numpy arrays of shape (D, L_i) where D is the number of features
                and L_i is the number of letters in the i-th example.
            y_test: list of length M_test of labels (strings)

        Returns:
            X_train: numpy array of shape (N_train, D + 1) where N_train is M_train * sum(L_i)
            y_train: numpy array of shape (N_train)
            X_test: numpy array of shape (N_test, D + 1) where N_test is M_test * sum(L_i)
            y_test: numpy array of shape (N_test)
        """

        self.X_train = np.concatenate(X_train, axis=1).transpose()
        self.X_test = np.concatenate(X_test, axis=1).transpose()

        # Add ones for bias
        self.X_train = np.concatenate([self.X_train, np.ones((self.X_train.shape[0], 1))], axis=1)
        self.X_test = np.concatenate([self.X_test, np.ones((self.X_test.shape[0], 1))], axis=1)

        # map characters to integers
        self.y_train = np.array([], dtype=np.uint8)
        for i in range(len(y_train)):
            chars = list(y_train[i])
            chars_ = np.array([self.mapping['characters'][chars[j]] for j in range(len(chars))], dtype=np.uint8)
            self.y_train = np.concatenate([self.y_train, chars_])

        self.y_test = np.array([], dtype=np.uint8)
        for i in range(len(y_test)):
            chars = list(y_test[i])
            chars_ = np.array([self.mapping['characters'][chars[j]] for j in range(len(chars))], dtype=np.uint8)
            self.y_test = np.concatenate([self.y_test, chars_])

    def train(self):
        counter = 0
        while True:
            end = True
            for i in range(self.num_train_samples):
                y_pred = self.predict(self.X_train[i])
                y_true = self.y_train[i]
                if y_pred != y_true:
                    self.W[:, y_true] += self.X_train[i]
                    self.W[:, y_pred] -= self.X_train[i]
                    end = False
            counter += 1
            if counter % 10 == 0:
                print(f"Train accuracy: {self.evaluate_train()}")
            if end:
                break
        print(f"Training completed with train accuracy: {self.evaluate_train()}")

    def predict(self, x: np.ndarray) -> np.ndarray:
        scores = np.dot(x, self.W)
        return np.argmax(scores)

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