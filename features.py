import pdb
import mmh3
import numpy as np
from numpy.linalg import norm
from scipy import sparse as sp
from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize
from sklearn.utils import check_array, check_random_state
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import check_is_fitted
from nlp import hashed_bongs, remove_punctuation, remove_stopwords, split_on_punctuation, word_tokenize
from utils import INT


TOKENIZER = {'simple': remove_punctuation,
             'custom': split_on_punctuation,
             'nltk': word_tokenize}


class RandomFourierFeaturizer:

    def __init__(self, n_components=1000, random_state=None, feature_range=(0, 1), copy=True, with_mean=True, with_std=True):
        '''RF preprocessing and featurizing
        Args:
            n_components: number of RF features
            random_state: passed to sklearn.utils.check_random_state
            feature_range: passed to sklearn.preprocessing.MinMaxScaler
            copy: passed to sklearn.preprocessing.MinMaxScaler and sklearn.preprocessing.StandardScaler
            with_mean: passed to sklearn.preprocessing.StandardScaler
            with_std: passed to sklearn.preprocessing.StandardScaler
        '''

        self.n_components = n_components
        self.random_state = random_state
        self._minmax = MinMaxScaler(feature_range=feature_range, copy=copy)
        self._standard = StandardScaler(copy=copy, with_mean=with_mean, with_std=with_std)

    def fit(self, X, y=None, **kwargs):
        '''computes normalization statistics and initializes RF featurization
        Args:
            X: training data
            y: passed to sklearn.preprocessing.MinMaxScaler.fit and sklearn.preprocessing.StandardScaler.fit
            kwargs: ignored
        Returns:
            self
        '''

        X = check_array(X, accept_sparse='csr')
        random_state = check_random_state(self.random_state)
        n_features = X.shape[1]

        self._minmax.fit(X, y=y)
        self._standard.fit(X, y=y)

        self.gaussian_weights_ = random_state.normal(size=(n_features, self.n_components)).astype(X.dtype)
        self.cauchy_weights_ = random_state.standard_cauchy(size=(n_features, self.n_components)).astype(X.dtype)
        self.random_offset_ = random_state.uniform(0, 2*np.pi, size=self.n_components).astype(X.dtype)

        return self

    def transform(self, X, preprocess=None, kernel='gaussian', gamma=1.0, **kwargs):
        '''preprocess and featurizes data
        Args:
            X: data to transform
            preprocess: which normalization to use (None|'minmax'|'normalize'|'standard')
            kernel: which kernel function to use ('gaussian'|'laplacian')
            gamma: bandwidth parameter (positive float)
            kwargs: ignored
        Returns:
            numpy.ndarray of shape (X.shape[0], self.n_components)
        '''

        check_is_fitted(self, 'random_offset_')

        if preprocess is None:
            X = check_array(X, accept_sparse='csr')
        elif preprocess == 'minmax':
            X = self._minmax.transform(X)
        elif preprocess == 'normalize':
            X = normalize(X)
        elif preprocess == 'standard':
            X = self._standard.transform(X)
        else:
            raise(NotImplementedError)

        if kernel == 'gaussian':
            projection = safe_sparse_dot(X, np.sqrt(2.0*gamma) * self.gaussian_weights_)
        elif kernel == 'laplacian':
            projection = safe_sparse_dot(X, gamma * self.cauchy_weights_)
        else:
            raise(NotImplementedError)

        projection += self.random_offset_
        np.cos(projection, out=projection)
        projection *= np.sqrt(2.0 / self.n_components)

        return projection

    def fit_transform(self, X, y=None, cfg2idx=None, **kwargs):
        '''preprocess and featurize data
        Args:
            X: data to transform
            y: ignored and returned
            cfg2idx: iterator with elements (config dict, array index)
            kwargs: passed to self.transform
        Returns:
            numpy.ndarray of shape (X.shape[0], self.n_components), numpy.ndarray of shape (X.shape[0],)
        '''

        if cfg2idx is None:
            return self.transform(X, **kwargs), y
        
        F = np.empty((len(X), self.n_components), dtype=X.dtype)
        for cfg, idx in cfg2idx:
            if idx.any():
                F[idx] = self.transform(X[idx], **cfg, **kwargs) 
        return F, y


def subdict(d, keys):

    return {key: d[key] for key in keys if key in d}


class HashedBonGFeaturizer:

    def __init__(self, n_components=1000, random_state=None, randomize=True, **kwargs):
        '''text preprocessing and BonG hashing
        Args:
            n_components: number of hash bins
            random_state: passed to sklearn.utils.check_random_state
            kwargs: ignored
        '''
        self.randomize = randomize

        self.n_components = n_components
        self.random_state = random_state

    def fit(self, *args, **kwargs):
        '''sets random seed
        Args:
           args: ignored
           kwargs: ignored
        Returns:
            self
        '''

        random_state = check_random_state(self.random_state)
        self.seed = random_state.randint(np.iinfo(INT).max)
        return self

    def _featurize(self, B, weights=None, alpha=1.0, preprocess=None):

        if preprocess == 'average':
            counts = np.array(B.sum(1))[:,0]
            counts[counts == 0.0] = 1.0

        if weights == 'nb':
            p, q = self.p + alpha, self.q + alpha
            p /= norm(p, 1)
            q /= norm(q, 1)
            B = B.dot(sp.diags(np.log2(p / q), 0))
        elif weights == 'sif':
            B = B.dot(sp.diags(self.total * alpha / (self.total * alpha + self.counts), 0))
        elif not weights is None:
            raise(NotImplementedError)

        if preprocess == 'average':
            B = sp.diags(1.0 / counts, 0).dot(B)
        elif preprocess == 'normalize':
            B = normalize(B, copy=False)
        elif not preprocess is None:
            raise(NotImplementedError)

        return B

    def transform(self, X, method='custom', stop=True, lower=True, order=1, binary=True, weights=None, alpha=1.0, preprocess=None, name=-1, **kwargs):
        '''preprocess and featurizes data
        Args:
            X: data to transform
            method: tokenization method ('simple'|'custom'|'nltk')
            stop: remove stopword tokens
            lower: lowercase tokens
            order: n-gram model order
            binary: binarize hashed features
            weights: feature weighting to use (None|'nb'|'sif')
            alpha: smoothing constant for feature weighting
            preprocess: which normalization to use (None|'average'|'normalize')
            name: modifies hash function seed
            kwargs: ignored
        Returns:
            numpy.ndarray of shape (X.shape[0], self.n_components)
        '''

        check_is_fitted(self, 'seed')

        if method == 'custom':
            tokenize = split_on_punctuation
        elif method == 'nltk':
            tokenize = word_tokenize
        elif method == 'simple':
            tokenize = remove_punctuation
        else:
            raise(NotImplementedError)

        if stop:
            if lower:
                docs = [[token.lower() for token in remove_stopwords(tokenize(doc))] for doc in X]
            else:
                docs = [list(remove_stopwords(tokenize(doc))) for doc in X]
        else:
            if lower:
                docs = [[token.lower() for token in tokenize(doc)] for doc in X]
            else:
                docs = [list(tokenize(doc)) for doc in X]

        if self.randomize:
            hash_seed = name + self.seed
        else:
            hash_seed = self.seed

        hash_func = lambda string: mmh3.hash(string, hash_seed, signed=False)
        B = hashed_bongs(docs, hash_func, self.n_components, order=order)
        if binary:
            B = B.sign()
        return self._featurize(B, weights=weights, alpha=alpha, preprocess=preprocess)

    def fit_transform(self, X, y=None, cfg2idx=None, **kwargs):
        '''preprocess and featurize data
        Args:
            X: data to transform (numpy.ndarray of text documents)
            y: data labels
            cfg2idx: iterator with elements (config dict, array index)
            kwargs: passed to self.transform
        Returns:
            numpy.ndarray of shape (X.shape[0], self.n_components), numpy.ndarray of shape (X.shape[0],)
        '''

        assert not y is None, "must provide targets"
        assert set(y) == {-1.0, 1.0}, "targets must be binary -1.0, 1.0"

        keys = ['method', 'lower', 'stop', 'order', 'binary', 'name']
        if cfg2idx is None:
            Flist, ylist, cfgs = [self.transform(X, **subdict(kwargs, keys))], [y], [kwargs]
        else:
            Flist, ylist, cfgs = zip(*((self.transform(X[idx], **subdict(cfg, keys)), y[idx], cfg) for cfg, idx in cfg2idx if idx.any()))

        self.p = sum(np.array(F[y == 1.0].sum(0))[0] for F, y in zip(Flist, ylist))
        self.q = sum(np.array(F[y == -1.0].sum(0))[0] for F, y in zip(Flist, ylist))
        self.counts = sum(np.array(F.sum(0))[0] for F in Flist)
        self.total = self.counts.sum()

        keys = ['weights', 'alpha', 'preprocess']
        return sp.vstack([self._featurize(F, **subdict(cfg, keys)) for F, cfg in zip(Flist, cfgs)]), np.hstack(ylist)
