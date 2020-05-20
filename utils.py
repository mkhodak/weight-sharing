import h5py
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import LinearSVC

FLOAT = np.float32
INT = np.int32


class SVMClassifier(LinearSVC):

    def __init__(self, **kwargs):

        super(SVMClassifier, self).__init__(**kwargs)

    def accuracy(self, X, labels, mean=True):

        if mean:
            return self.score(X, 2.0 * labels - 1.0)
        return self.predict(X) == 2.0 * labels - 1.0

class LogitClassifier(LogisticRegression):

    def __init__(self, **kwargs):

        super(LogitClassifier, self).__init__(**kwargs)

    def accuracy(self, X, labels, mean=True):

        if mean:
            return self.score(X, 2.0 * labels - 1.0)
        return self.predict(X) == 2.0 * labels - 1.0


class RidgeClassifier(Ridge):
    #NOTE: for best results, encode labels as a multi-column {-1.0, 1.0} matrix before passing to the 'y' argument of sklearn.linear_model.Ridge.fit 

    def __init__(self, **kwargs):
        '''wrapper on top of sklearn.linear_model.Ridge to enable multi-class classification because sklearn.linear_model.RidgeClassifier inexplicably cannot handle it in certain versions of scikit-learn 
        Args:
            kwargs: passed to sklearn.linear_model.Ridge
        '''

        super(RidgeClassifier, self).__init__(**kwargs)

    def accuracy(self, X, labels, mean=True):
        '''computes accuracy
        Args:
            X: features
            labels: vector of length X.shape[0] with numerical class labels
            mean: take mean (otherwise return Boolean vector of length X.shape[0])
        Returns:
            0-1 accuracy (float)
        '''

        accs = (X.dot(self.coef_.T)+self.intercept_).argmax(1) == labels
        if mean:
            return np.mean(accs)
        return accs


class ArrayStore(h5py.File):

    def __init__(self, ram, numel, *args, **kwargs):
        '''divides storage of identical-size arrays between RAM and disc
        Args:
            ram: maximum RAM to use, in GB; note this only limits the memory usage of this object
            numel: number of elements in each matrix
            args: passed to h5py.File
            kwargs: passed to h5py.File
        '''
        
        super(ArrayStore, self).__init__(*args, **kwargs)
        self.max_in_ram = int(ram * 8E9 / numel / 32)
        self.cache = {}
        self.names = set()

    def add(self, name, data):
        '''store named array:
        Args:
            name: name of array
            data: array
        '''

        if len(self.cache) < self.max_in_ram:
            self.cache[name] = data
        else:
            self.create_dataset(name, data=data)
        self.names.add(name)
        
    def get(self, name):
        '''retrieve named array
        Args:
            name: name of array
        Returns:
            array
        '''

        try:
            return self.cache[name]
        except KeyError:
            output = super(ArrayStore, self).get(name)[()]
            if len(self.cache) < self.max_in_ram:
                self.cache[name] = output
                del self[name]
            return output

    def remove(self, name):
        '''remove named array from storage
        Args:
            name: name of array
        '''

        try:
            del self.cache[name]
        except KeyError:
            del self[name]
        self.names.remove(name)
