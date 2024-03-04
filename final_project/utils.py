import numpy as np
from keras.utils import to_categorical


def init_data(subject=None, verbose=False):
    """
    Training/Valid data shape: (2115, 22, 1000), float tensor
    Training/Valid target shape: (2115,), (Int tensor with values in [769, 770, 771, 772])
    Test data shape: (443, 22, 1000)
    Test target shape: (443,)
    Person train/valid shape: (2115, 1), [0, 0, 0, 0, 0, 1, 1, ... 8, 8]
    Person test shape: (443, 1), [0, 0, 0, 0, 0, 1, 1, ... 8, 8]
    """
    X_train_valid = np.load("X_train_valid.npy")
    y_train_valid = np.load("y_train_valid.npy") - 769
    X_test = np.load("X_test.npy")
    y_test = np.load("y_test.npy") - 769
    person_train_valid = np.load("person_train_valid.npy")
    person_test = np.load("person_test.npy")
    if subject is None:
        return X_train_valid, y_train_valid, X_test, y_test

    X_train_subject = X_train_valid[np.where(person_train_valid == subject)[0]]
    y_train_subject = y_train_valid[np.where(person_train_valid == subject)[0]]
    X_test_subject = X_test[np.where(person_test == subject)[0]]
    y_test_subject = y_test[np.where(person_test == subject)[0]]
    if verbose:
        print("Training Subject {}:".format(subject), end=" ")
        print("Training data shape: {}".format(X_train_subject.shape), end=" ")
        print("Testing data shape: {}".format(y_train_subject.shape))
        print("Testing Subject {}:".format(subject), end=" ")
        print("Training data shape: {}".format(X_test_subject.shape), end=" ")
        print("Testing data shape: {}".format(y_test_subject.shape))
    return X_train_subject, y_train_subject, X_test_subject, y_test_subject


def preprocess_data(X_train, y_train, X_test, y_test, verbose=False):
    X_train = X_train[:, :, :500]
    X_test = X_test[:, :, :500]
    X__train_max = np.max(X_train.reshape(X_train.shape[0], X_train.shape[1], -1, 2), axis=3)
    X_test_max = np.max(X_test.reshape(X_test.shape[0], X_test.shape[1], -1, 2), axis=3)
    X_average_train = np.mean(X_train.reshape(X_train.shape[0], X_train.shape[1], -1, 2), axis=3)
    X_average_train = X_average_train + np.random.normal(0.0, 0.5, X_average_train.shape)
    total_X_train = np.vstack((X__train_max, X_average_train))
    total_y_train = np.hstack((y_train, y_train))
    for i in range(2):
        X_subsample = X_train[:, :, i::2] + np.random.normal(0.0, 0.5, X_train[:, :, i::2].shape)

        total_X_train = np.vstack((total_X_train, X_subsample))
        total_y_train = np.hstack((total_y_train, y_train))

    if verbose:
        print("Training data shape: {}".format(total_X_train.shape), end=" ")
        print("Training target shape: {}".format(total_y_train.shape))
        print("Testing data shape: {}".format(X_test_max.shape), end=" ")
        print("Testing target shape: {}".format(y_test.shape))

    return total_X_train, total_y_train, X_test_max, y_test


def window_data(X_train, y_train, window_size=200, stride=100, verbose=False):
    num_samples, num_features, num_timepoints = X_train.shape
    num_windows = (num_timepoints - window_size) // stride
    X_train_window = np.zeros((num_samples * num_windows, num_features, window_size))
    y_train_window = np.zeros((num_samples * num_windows,))

    for i in range(num_samples):
        for j in range(num_features):
            for k in range(num_windows):
                X_train_window[i * num_windows + k, j, :] = X_train[i, j, k * 100:k * 100 + window_size]
                y_train_window[i * num_windows + k] = y_train[i]

    if verbose:
        print("WTraining data shape: {}".format(X_train_window.shape), end=" ")
        print("WTraining target shape: {}".format(y_train_window.shape))

    return X_train_window, y_train_window



def load_data(X_train, y_train, X_test, y_test, verbose=False):

    ind_valid = np.random.choice(X_train.shape[0], int(0.2 * X_train.shape[0]), replace=False)
    ind_train = np.array(list(set(range(X_train.shape[0])).difference(set(ind_valid))))
    (x_train, x_valid) = X_train[ind_train], X_train[ind_valid]
    (y_train, y_valid) = y_train[ind_train], y_train[ind_valid]

    y_train = to_categorical(y_train, 4)
    y_valid = to_categorical(y_valid, 4)

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_valid = x_valid.reshape(x_valid.shape[0], x_valid.shape[1], x_train.shape[2], 1)

    x_train = np.swapaxes(x_train, 1, 3)
    x_train = np.swapaxes(x_train, 1, 2)
    x_valid = np.swapaxes(x_valid, 1, 3)
    x_valid = np.swapaxes(x_valid, 1, 2)

    x_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
    x_test = np.swapaxes(x_test, 1, 3)
    x_test = np.swapaxes(x_test, 1, 2)
    y_test = to_categorical(y_test, 4)

    if verbose:
        print("Training data shape: {}".format(x_train.shape), end=" ")
        print("Training target shape: {}".format(y_train.shape))
        print("Validation data shape: {}".format(x_valid.shape), end=" ")
        print("Validation target shape: {}".format(y_valid.shape))
        print("Testing data shape: {}".format(x_test.shape), end=" ")
        print("Testing target shape: {}".format(y_test.shape))

    # 5 folds including validation
    # X_tensor_train = torch.tensor(X_train, dtype=torch.float32)
    # y_tensor_train = torch.tensor(y_train, dtype=torch.long)
    # X_tensor_test = torch.tensor(X_test, dtype=torch.float32)
    # y_tensor_test = torch.tensor(y_test, dtype=torch.long)

    # train_set = TensorDataset(X_tensor_train, y_tensor_train)
    # train_subset, val_subset = torch.utils.data.random_split(train_set,
    #                                                          [int(0.8 * X_train.shape[0]), int(0.2 * X_train.shape[0])],
    #                                                          generator=torch.Generator().manual_seed(1))
    #
    # train_loader = torch.utils.data.DataLoader(train_subset, shuffle=True, batch_size=32)
    # val_loader = torch.utils.data.DataLoader(val_subset, shuffle=False, batch_size=32)
    # test_loader = torch.utils.data.DataLoader(TensorDataset(X_tensor_test, y_tensor_test))
    # return train_loader, val_loader, test_loader
    return x_train, y_train, x_valid, y_valid, x_test, y_test
