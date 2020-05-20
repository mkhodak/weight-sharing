import argparse
import random
import time
import gc
import os
import numpy as np
from tempfile import TemporaryFile
import h5py
from cifar import load as cifar
from features import HashedBonGFeaturizer as HBF
from features import RandomFourierFeaturizer as RFF
from imdb import load as imdb
from utils import ArrayStore, LogitClassifier, RidgeClassifier, SVMClassifier
from hyper import OPTIONS, grid_configs, random_configs

def parse():

    parser = argparse.ArgumentParser('hyperband for hyperparameter optimization')
    parser.add_argument('dataset', type=str, help='cifar or imdb')
    parser.add_argument('--min_features', type=int, default=10, help='minimum number of RF features')
    parser.add_argument('--features', type=int, default=100, help='maximum number of RF features')
    parser.add_argument('--eta', type=int, default=2, help='reduction factor')
    parser.add_argument('--s_run', type=int, default=None, help='option to repeat a specific bracket')
    parser.add_argument('--grid', action='store_true', help='use parameter grid instead of random parameters')
    parser.add_argument('--search', type=int, default=32, help='size of search space')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--alpha', type=float, default=0.5, help='Ridge regularization coefficient')
    parser.add_argument('--coef', type=float, default=1.0, help='Logit regularization coefficient')
    parser.add_argument('--exact', action='store_true', help='compute exact val acc for each config')
    parser.add_argument('--svm', action='store_true', help='use SVM for IMDB')

    return parser

#RAM = 200
RAM = 1
FIT_T_TOAL = 0.0

class Hyperband():
    def __init__(self, args, configs):
        self.args = args
        self.configs = configs

        if self.args.dataset == 'cifar':
            load = cifar
        elif self.args.dataset == 'imdb':
            load = imdb
        else:
            raise(NotImplementedError)

        self.Xtrain, self.Ytrain, self.Ltrain = load('train')
        self.Xval, self.Yval, self.Lval = load('val')
        self.Xtest, self.Ytest, self.Ltest = load('test')

    def get_hyperparameter_configuration(self, n):
        ''' Uniformly samples n hyperparameter configurations
        Args:
            n: number of samples
        Returns:
            list of configurations
        '''
        return list(np.random.choice(self.configs, int(n)))

    def run_then_return_val_acc(self, config, r):
        ''' Train the configuration t with r resources 
        Args:
            config: configuration to train with
            r: resources allocated
        Returns:
            validation accuracy given t, r
        '''

        t = time.time()

        if self.args.dataset == 'cifar':
            classifier = lambda: RidgeClassifier(alpha=self.args.alpha, solver='lsqr', copy_X=False, random_state=args.seed)
            featurizer = RFF(n_components=int(np.ceil(r)))
        elif self.args.dataset == 'imdb':
            if args.svm:
                print('[svm] using SVM classifier...')
                classifier = lambda: SVMClassifier(random_state=args.seed, C=args.coef, tol=1e-5)
            else:
                classifier = lambda: LogitClassifier(C=args.coef, solver='liblinear', random_state=args.seed)
            featurizer = HBF(n_components=int(np.ceil(r)))
        else:
            raise(NotImplementedError)

        Xtrain, Ytrain, Ltrain = self.Xtrain, self.Ytrain, self.Ltrain
        Xval, Yval, Lval = self.Xval, self.Yval, self.Lval
        Xtest, Ytest, Ltest = self.Xtest, self.Ytest, self.Ltest

        global FIT_T_TOTAL
        FIT_T_TOTAL = 0.0
        #with TemporaryFile(dir=os.path.join(os.getcwd(), 'kernel')) as tf:
        #    with ArrayStore(RAM, len(Xval) * featurizer.n_components, tf) as store:

        featurizer.fit(Xtrain)

        print('Mapping', end='\r')
        FYtrain = featurizer.fit_transform(Xtrain, y=Ytrain, **config)

        print('Fitting', end='\r')
        clf = classifier()
        fit_t_start = time.time()
        clf.fit(*FYtrain)
        fit_t_end = time.time() - fit_t_start
        FIT_T_TOTAL += fit_t_end

        print('Scoring', end='\r')
        # TODO try array store if memory issues
        Fval = featurizer.transform(Xval, **config)
        score = clf.accuracy(Fval, Lval)

        ### EVAL ### time is excluded
        end_eval = 0
        start_eval = time.time()

        # Get validation score
        Fval = featurizer.transform(Xval, **config)
        score = clf.accuracy(Fval, Lval)

        # Get test score
        Ftest = featurizer.transform(Xtest, **config)
        score_test = clf.accuracy(Ftest, Ltest)

        # Train oracle
        #if self.args.dataset == 'cifar':
        #    oracle_fzr = RFF(n_components=int(np.ceil(r)))
        #elif self.args.dataset == 'imdb':
        #    oracle_fzr = HBF(n_components=int(np.ceil(r)))
        #else:
        #    raise(NotImplementedError)
        #oracle_fzr.fit(Xtrain)
        #FYtrain = oracle_fzr.fit_transform(Xtrain, y=Ytrain, **config)
        #clf = classifier()
        #clf.fit(*FYtrain)

        # Get oracle validation score
        #Fval = oracle_fzr.transform(Xval, **config)
        #score_oracle = score clf.accuracy(Fval, Lval)

        # Get oracle test score
        #Ftest = oracle_fzr.transform(Xtest, **config)
        #score_test_oracle = score_test clf.accuracy(Ftest, Ltest)

        end_eval = time.time() - start_eval
        ### end EVAL ### time is excluded

        watch = time.time() - t - end_eval

        print('\tVal:', round(score, 5),
                      '\tTest:', round(score_test, 5),
                      '\tO-Val:', round(score, 5),
                      '\tO-Test:', round(score_test, 5),
                      '\tTime:', round(watch), 'seconds',
                      '\tConfig:', config)

        gc.collect()

        return score

    def top_k(self, configs, accs, k):
        ''' Computes top k performing configurations
        Args:
            configs: list of configurations
            accs: list of accuracies
            k: the number of configs to return
        Returns:
            top k configs according to accuracy
        '''

        config_perfs = zip(accs, configs)
        sorted_config_perfs = sorted(config_perfs, key=lambda x: x[0])[::-1]
        sorted_perfs = [p for p, _ in sorted_config_perfs]
        sorted_configs = [c for _, c in sorted_config_perfs]

        return sorted_configs[:int(k)], sorted_perfs[:int(k)]

    def run(self, R=10000, r=100, eta=3, s_run=None):
        def log_eta(x):
            return np.round(np.log(x) / np.log(eta), decimals=10)

        best = None
        best_acc = 0
        s_max = np.floor(log_eta(R / r))
        B = int(s_max + 1) * R

        assert((s_run == None) or ((0 <= s_run) and (s_run <= s_max)))

        for ell in np.flip(np.arange(s_max + 1), axis=0):

            n = int((B * (eta ** ell)) / (R * (ell + 1)))

            if n > 0:

                s = 0
                while n * R * (s + 1.) * eta ** (-s) > B:
                    s += 1

                if s_run is None or s == s_run:

                    print()
                    print('s=%d, n=%d' % (s, n))
                    print('n_i\tr_k')

                    # begin SuccessiveHalving with (n, r) inner loop
                    # TODO feed in n explicitly (and max resource r to back into min resources)
                    # TODO just feed in all 32 configs
                    T = self.get_hyperparameter_configuration(n)

                    for i in range(int(s) + 1):
                        i = np.float(i)
                        n_i = np.floor(n * (eta ** -i))
                        r_i = R * (eta ** (i - s))

                        print('%d\t%d' %(n_i, r_i))

                        L = [self.run_then_return_val_acc(t, r_i) for t in T]
                        T, T_accs = self.top_k(T, L, np.floor(n_i / eta))

                        if len(T) > 0:
                            if best_acc <= T_accs[0]:
                                best = T[0]
                                best_acc = T_accs[0]

        # Return config with lowest intermediate loss seen thus far
        print("Best config:", self.run_then_return_val_acc(best, R))
        print("Intermediate val acc:", best_acc)
        print("Final val acc:", self.run_then_return_val_acc(best, R))
        print('Total fit time\t', FIT_T_TOTAL, 'seconds')
        return best, best_acc

if __name__ == "__main__":
    args = parse().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.grid:
        configs = grid_configs(OPTIONS[args.dataset])
    else:
        configs = random_configs(OPTIONS[args.dataset], args.search, continuous={'alpha', 'gamma'})
    for i, config in enumerate(configs):
        config['name'] = i

    hyperband = Hyperband(args, configs)
    best, best_acc = hyperband.run(R=args.features, r=args.min_features, eta=args.eta,
                  s_run=args.s_run)

