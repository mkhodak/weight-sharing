import random
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from torchvision import datasets, transforms
from features import RandomFourierFeaturizer as RFF
from utils import FLOAT, INT, RidgeClassifier


NUM_CLASS = 10
NUM_TRAIN = 40000
NUM_VALID = 10000


def load(partition='train'):
    '''loads CIFAR-10 data
    Args:
        partition: data to load ('train'|'val'|'test')
    Returns:
        data array, array of {-1,1}-label encodings, label vector
    '''

    data = datasets.CIFAR10(root='kernel/data', train=partition != 'test', download=True, transform=transforms.ToTensor())

    if partition == 'train':
        X = np.stack([data[i][0].flatten().numpy() for i in range(NUM_TRAIN)])
        L = np.fromiter((data[i][1] for i in range(NUM_TRAIN)), dtype=INT, count=NUM_TRAIN)
    else:
        X = np.stack([data[i][0].flatten().numpy() for i in range(-NUM_VALID, 0)])
        L = np.fromiter((data[i][1] for i in range(-NUM_VALID, 0)), dtype=INT, count=NUM_VALID)
    Y = 2.0 * np.array(OneHotEncoder(categories='auto').fit_transform(L[:,None]).todense(), dtype=FLOAT) - 1.0
    return X, Y, L


def test():

    random.seed(0)
    np.random.seed(0)
    rff = RFF(n_components=10000)
    config = {'preprocess': 'standard', 'kernel': 'gaussian', 'gamma': 5E-4}

    Xtrain, Ytrain, Ltrain = load()
    print('Featurizing data')
    rff.fit(Xtrain)
    Ftrain, _ = rff.fit_transform(Xtrain, **config)

    print('Fitting model')
    clf = RidgeClassifier(alpha=0.5, solver='lsqr', copy_X=True)
    clf.fit(Ftrain, Ytrain)
    print('Train Acc:', clf.accuracy(Ftrain, Ltrain))

    Xval, _, Lval = load('val')
    Fval = rff.transform(Xval, **config)
    print('Val Acc:', clf.accuracy(Fval, Lval))

    Xtest, _, Ltest = load('test')
    Ftest = rff.transform(Xtest, **config)
    print('Test Acc:', clf.accuracy(Ftest, Ltest))


if __name__ == '__main__':

    test()
