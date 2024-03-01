import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, IterableDataset


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


def load_data(X_train, y_train, X_test, y_test, verbose=False):
    # feature scaling
    X_train -= np.mean(X_train, axis=0)
    X_train /= np.std(X_train, axis=0)
    X_test -= np.mean(X_train, axis=0)
    X_test /= np.std(X_train, axis=0)

    # 5 folds including validation
    train_subset, val_subset = torch.utils.data.random_split(X_train, [int(0.8 * X_train.shape[0]), int(0.2 * X_train.shape[0])],
                                                             generator=torch.Generator().manual_seed(1))
    X_new_train = X_train[train_subset.indices]
    y_new_train = y_train[train_subset.indices]

    X_new_val = X_train[val_subset.indices]
    y_new_val = y_train[val_subset.indices]

    if verbose:
        print("X_new_train shape: {}".format(X_new_train.shape))
        print("y_new_train shape: {}".format(y_new_train.shape))
        print("X_new_val shape: {}".format(X_new_val.shape))
        print("y_new_val shape: {}".format(y_new_val.shape))

    X_tensor_train = torch.tensor(X_new_train, dtype=torch.float32)
    y_tensor_train = torch.tensor(y_new_train, dtype=torch.long)
    train_set = IterableDataset(X_tensor_train, y_tensor_train)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

    return train_loader


X_train, y_train, X_test, y_test = init_data(subject=None, verbose=True)
train_loader = load_data(X_train, y_train, X_test, y_test, verbose=True)
print(train_loader)
