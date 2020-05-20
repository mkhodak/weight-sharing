import csv
import os
import pdb
import random
import numpy as np
import wget
from features import HashedBonGFeaturizer as HBF
from utils import FLOAT, INT, LogitClassifier


NUM_VALID = 12500
SHUFFLE_SEED = 0


def consistent_shuffle(*args, **kwargs):

    state = random.getstate()
    for arg in args:
        random.seed(**kwargs)
        random.shuffle(arg)
    random.setstate(state)


def load(partition='train'):

    if partition == 'train':
        if os.path.isfile('kernel/data/imdb_train.csv'):
            print('File already downloaded')
        else:
            print('Downloading IMDB dataset')
            wget.download('https://raw.githubusercontent.com/NLPrinceton/text_embedding/master/data-documents/imdb_train.csv', 'kernel/data/imdb_train.csv', bar=None)
        with open('kernel/data/imdb_train.csv', 'r') as f:
            docs, labels = list(zip(*((document, label.split(':')[0]) for label, document in csv.reader(f, delimiter='\t'))))
        docs, labels = list(docs), list(labels)
        consistent_shuffle(docs, labels, a=SHUFFLE_SEED)
        offset = 0

    else:
        if os.path.isfile('kernel/data/imdb_test.csv'):
            print('File already downloaded')
        else:
            wget.download('https://raw.githubusercontent.com/NLPrinceton/text_embedding/master/data-documents/imdb_test.csv', 'kernel/data/imdb_test.csv', bar=None)
        with open('kernel/data/imdb_test.csv', 'r') as f:
            docs, labels = list(zip(*((document, label.split(':')[0]) for label, document in csv.reader(f, delimiter='\t'))))
        docs, labels = list(docs), list(labels)
        consistent_shuffle(docs, labels, a=SHUFFLE_SEED)
        docs, labels = (docs[:NUM_VALID], labels[:NUM_VALID]) if partition == 'test' else (docs[NUM_VALID:], labels[NUM_VALID:])
        offset = 37500 if partition == 'test' else 25000

    L = np.fromiter((1 if label == 'pos' else 0 for label in labels), dtype=INT, count=len(labels))
    return np.array(list(zip((str(i+offset) for i in range(len(docs))), docs))), 2.0 * L.astype(FLOAT) - 1.0, L


def test():

    random.seed(0)
    np.random.seed(0)
    hbf = HBF(n_components=1024*1024)
    config = {'method': 'nltk', 'stop': False, 'lower': False, 'order': 2, 'binary': True, 'weights': 'nb', 'alpha': 1.0, 'preprocess': None}

    Xtrain, Ytrain, Ltrain = load('train')
    print('Featurizing data')
    hbf.fit()
    Ftrain, Ytrain = hbf.fit_transform(Xtrain, y=Ytrain, **config)

    print('Fitting model')
    clf = LogitClassifier(solver='liblinear')
    clf.fit(Ftrain, Ytrain)
    print('Train Acc:', clf.accuracy(Ftrain, Ltrain))

    Xval, _, Lval = load('val')
    Fval = hbf.transform(Xval, **config)
    print('Val Acc:', clf.accuracy(Fval, Lval))

    Xtest, _, Ltest = load('test')
    Ftest = hbf.transform(Xtest, **config)
    print('Test Acc:', clf.accuracy(Ftest, Ltest))


if __name__ == '__main__':

    test()
