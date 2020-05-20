import argparse
import gc
import os
import pdb
import pickle
import random
import time
from itertools import product
from operator import itemgetter
from tempfile import TemporaryFile
import h5py
import numpy as np
from kernel.cifar import load as cifar
from kernel.features import HashedBonGFeaturizer as HBF
from kernel.features import RandomFourierFeaturizer as RFF
from kernel.imdb import load as imdb
from kernel.utils import ArrayStore, LogitClassifier, RidgeClassifier


OPTIONS = {'cifar': {'preprocess': [None, 'minmax', 'normalize', 'standard'],
                     'kernel': ['gaussian', 'laplacian'],
                     'gamma': [10 ** loggamma for loggamma in range(-5, 2)]},
           'imdb': {'method': ['simple', 'custom', 'nltk'],
                    'stop': [True, False],
                    'lower': [True, False],
                    'order': list(range(1, 4)),
                    'binary': [True, False],
                    'weights': ['nb', 'sif'], 
                    'alpha': [10 ** logalpha for logalpha in range(-5, 2)],
                    'preprocess': [None, 'normalize', 'average']}}
RAM = 200


def parse():

    parser = argparse.ArgumentParser('weight-sharing for hyperparameter optimization')
    parser.add_argument('dataset', type=str, help='cifar or imdb')
    parser.add_argument('--features', type=int, default=1000, help='number of RF features')
    parser.add_argument('--grid', action='store_true', help='use parameter grid instead of random parameters')
    parser.add_argument('--search', type=int, default=32, help='size of search space')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--sweep', action='store_true', help='run parameter sweep')
    parser.add_argument('--switch', type=int, default=0, help='switch to sweep when this many configs left')
    parser.add_argument('--alpha', type=float, default=0.5, help='Ridge regularization coefficient')
    parser.add_argument('--coef', type=float, default=1.0, help='Logit regularization coefficient')
    parser.add_argument('--exact', action='store_true', help='compute exact val acc for each config')
    parser.add_argument('--grad', action='store_true', help='use exponentiated gradient instead of successive elimination')
    parser.add_argument('--eta', type=float, default=5.0, help='learning rate for exponentiated gradient')
    parser.add_argument('--niter', type=int, default=5, help='number of iterations for exponentiated gradient')
    parser.add_argument('--baseline', action='store_true', help='use baseline when computing gradient')
    parser.add_argument('--keep', type=float, default=0.5, help='proportion of configurations to keep in successive elimination')
    parser.add_argument('--output', type=str, help='name of output dump file')

    return parser


def param_sweep(configs, classifier, featurizer, Xtrain, Ytrain, Ltrain, Xval, Yval, Lval, array_store=None, start_time=None, output=None, **kwargs):
    '''runs parameter sweep over list of configurations
    Args:
        configs: list of configuration dicts
        classifier: returns a sklearn classifier object
        featurizer: object from kernel.features
        Xtrain: training data
        Ytrain: training label encodings
        Ltrain: training labels
        Xval: validation data
        Yval: validation label encodings
        Lval: validation labels
        array_store: initialized kernel.utils.ArrayStore object
        start_time: start clock at this time
        output: output dict
        kwargs: ignored
    Returns:
        best validation accuracy, corresponding configuration, time spent
    '''

    if start_time is None:
        featurizer.fit(Xtrain)

    if not output is None:
        output.setdefault('time', [])
        output.setdefault('scores', [])
        best = 0.0

    t = time.time() if start_time is None else start_time
    scores = []
    for config in configs:

        print('Mapping', end='\r')
        FYtrain = featurizer.fit_transform(Xtrain, y=Ytrain, **config)

        print('Fitting', end='\r')
        clf = classifier()
        clf.fit(*FYtrain)

        print('Scoring', end='\r')
        Fval = featurizer.transform(Xval, **config) if array_store is None else array_store.get(config['name'])
        score = clf.accuracy(Fval, Lval)

        scores.append(score)
        watch = time.time() - t
        print('Val:', round(score, 5),
              '\tTime:', round(watch), 'seconds',
              '\tConfig:', config)
        if not output is None:
            output['time'].append(watch)
            if score >= best:
                best = score
                output['test'] = clf

        gc.collect()

    best, config = max(zip(scores, configs), key=itemgetter(0))
    if not output is None:
        featurizer.fit_transform(Xtrain, y=Ytrain, **config)
        output['scores'].extend(scores)
    return best, config, watch


def successive_elimination(clf, featurizer, Xval, Lval, dist, configs, keep=0.5, store=None, **kwargs):

    scores = []
    top = int(np.ceil(keep * len(configs)))
    if store is None:
        rand = np.random.random(len(Xval))
        lower = 0.0
    for prob, config in zip(dist, configs):
        if store is None:
            select = (rand >= lower) * (rand < lower+prob)
            if select.any():
                scores.append(clf.accuracy(featurizer.transform(Xval[select], **config), Lval[select]))
            else:
                scores.append(0.0)
            lower += prob
        else:
            name = config['name']
            if name in store.names:
                Fval = store.get(name)
            else:
                Fval = featurizer.transform(Xval, **config)
            scores.append(clf.accuracy(Fval, Lval))
            if not name in store.names and (len(scores) < top or scores[-1] >= sorted(scores)[-top]):
                store.add(name, Fval)

    support = sum(dist > 0.0)
    if support == 1:
        return dist, scores, configs
    quantile = np.quantile(np.fromiter((score for prob, score in zip(dist, scores) if prob), float, count=support), 1.0-keep)
    for i, score in enumerate(scores):
        if score < quantile:
            dist[i] = 0.0
    support = sum(dist > 0.0)
    if support > top:
        lowest = min(score for prob, score in zip(dist, scores) if prob)
        for i in random.sample([i for i, score in enumerate(scores) if score == lowest], support-top):
            dist[i] = 0.0
    dist /= dist.sum()
    if not store is None:
        for name in store.names.difference(config['name'] for prob, config in zip(dist, configs) if prob):
            store.remove(name)
    return dist, scores, configs


def exponentiated_gradient(clf, featurizer, Xval, Lval, dist, configs, eta=10.0, baseline=0.0, **kwargs):

    rand = np.random.random(len(Xval))
    lower = 0.0
    grad = np.zeros(dist.shape, dtype=dist.dtype)
    scores = []
    for i, (prob, config) in enumerate(zip(dist, configs)):
        select = (rand >= lower) * (rand < lower+prob)
        if select.any():
            scores.append(clf.accuracy(featurizer.transform(Xval[select], **config), Lval[select]))
            grad[i] = (1.0-scores[-1]-baseline) * select.sum() / len(Lval) / prob
        else:
            scores.append(0.0)
        lower += prob

    dist *= np.exp(-eta * grad)
    dist /= dist.sum()
    return dist, scores, configs


def random_assignment(n_assignments, dist):

    assignments = np.random.choice(np.arange(len(dist)), size=n_assignments, p=dist)
    return np.vstack([assignments == i for i in range(len(dist))])

def weight_sharing(configs, classifier, featurizer, Xtrain, Ytrain, Ltrain, Xval, Yval, Lval, grad=False, eta=10.0, niter=5, baseline=False, keep=0.5, switch_to_sweep=0, estimate_val=True, output=None, args=None):
    '''runs successive-halving + weight-sharing over list of configurations
    Args:
        configs: list of configuration dicts
        classifier: returns a sklearn classifier object
        featurizer: object from kernel.features
        Xtrain: training data
        Ytrain: training label encodings
        Ltrain: training labels
        Xval: validation data
        Yval: validation label encodings
        Lval: validation labels
        grad: use exponentiated gradient rather instead of successive elimination
        eta: learning rate of exponentiated gradient
        niter: number of iterations for exponentiated gradient
        baseline: use baseline when computing gradient
        keep: proportion of configurations to keep in successive elimination
        switch_to_sweep: run parameter sweep over remaining configurations when this many are left
        estimate_val: use subsampled estimate of validation accuracy at each stage
        output: output dict
        args: the command line arguments
    Returns:
        (best validation accuracy, corresponding configuration), time spent
    '''

    featurizer.fit(Xtrain)
    dist = np.ones(len(configs)) / len(configs)
    update_dist = exponentiated_gradient if grad else successive_elimination
    niter = niter if grad else int(np.ceil(np.log(len(configs)) / np.log(1.0 / keep)))
    baseline_avg = 0.0
    
    if not output is None:
        output['time'] = []
        output['scores'] = []
        output['dist'] = [np.copy(dist)]

    t = time.time()
    exclude_time = 0
    with TemporaryFile(dir=os.path.join(os.getcwd(), 'kernel')) as tf:
        with ArrayStore(RAM, len(Xval) * featurizer.n_components, tf) as store:
            for i in range(niter):

                print('Mapping', end='\r')
                FYtrain = featurizer.fit_transform(Xtrain, y=Ytrain, cfg2idx=zip(configs, random_assignment(len(Xtrain), dist)))

                print('Fitting', end='\r')
                clf = classifier()
                clf.fit(*FYtrain)

                print('Scoring', end='\r')
                dist, scores, configs = update_dist(clf, featurizer, Xval, Lval, dist, configs, eta=eta, keep=keep, store=None if estimate_val else store, baseline=baseline_avg)
                if baseline:
                    baseline_avg += (dist * (1.0-np.array(scores))).sum()
                    baseline_avg /= i + 1
                    
                best, config = max(zip(dist, scores, configs), key=itemgetter(0))[1:] if grad else max(zip(scores, configs), key=itemgetter(0))

                eval_start =  time.time()
                # TODO Evaluate on full validation set - from scratch with new weights
                if args.dataset == 'cifar':
                    featurizer_new = RFF(n_components=args.features)
                elif args.dataset == 'imdb':
                    featurizer_new = HBF(n_components=args.features)
                else:
                    raise(NotImplementedError)
                featurizer_new.fit(Xtrain)
                FYtrain = featurizer.fit_transform(Xtrain, y=Ytrain, cfg2idx=zip(configs, random_assignment(len(Xtrain), dist)))
                clf = classifier()
                clf.fit(*FYtrain)
                Fval = featurizer.transform(Xval, **config)
                best_full = clf.accuracy(Fval, Lval)

                # Time accounting. Exclude evaluation time. 
                exclude_time += time.time() - eval_start
                watch = time.time() - t
                watch -= exclude_time

                print('Val:', round(best_full, 5), # Reporting from-scratch training
                      '\tTime:', round(watch), 'seconds',
                      '\tConfig:', config)
                #print('Val:', round(best, 5),
                #      '\tTime:', round(watch), 'seconds',
                #      '\tConfig:', config)
                if not output is None:
                    output['time'].append(watch)
                    output['dist'].append(np.copy(dist))
                    output['scores'].append(np.array(scores))

                if sum(dist > 0.0) <= switch_to_sweep and not grad:
                    return param_sweep(configs, classifier, featurizer, Xtrain, Ytrain, Ltrain, Xval, Yval, Lval, array_store=None if estimate_val else store, start_time=t, output=output)

                gc.collect()

            if not niter:
                config = configs[0]
    
            print('Mapping', end='\r')
            FYtrain = featurizer.fit_transform(Xtrain, y=Ytrain, **config)

            print('Fitting', end='\r')
            clf = classifier()
            clf.fit(*FYtrain)

            print('Scoring', end='\r')
            Fval = featurizer.transform(Xval, **config) if estimate_val else store.get(config['name'])

            score = clf.accuracy(Fval, Lval)
            watch = time.time() - t
            if not output is None:
                output['time'].append(watch)
                output['scores'].append(np.fromiter((score if cfg['name'] == config['name'] else 0.0 for cfg in configs), float, count=len(configs)))
                output['test'] = clf
            return score, config, watch


def grid_configs(d):

    keys, values = zip(*sorted(d.keys(), key=itemgetter(0)))
    return [dict(zip(keys, opts)) for opts in product(*values)]

def random_configs(d, n_samples, continuous=set()):

    return [{key: 10 ** np.random.uniform(np.log10(min(value)), np.log10(max(value))) 
             if key in continuous else random.sample(value, 1)[0] 
             for key, value in sorted(d.items(), key=itemgetter(0))} for _ in range(n_samples)] 

def main():

    args = parse().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.dataset == 'cifar':
        load = cifar
        classifier = lambda: RidgeClassifier(alpha=args.alpha, solver='lsqr', copy_X=False, random_state=args.seed)
        featurizer = RFF(n_components=args.features)
    elif args.dataset == 'imdb':
        load = imdb
        classifier = lambda: LogitClassifier(C=args.coef, solver='liblinear', random_state=args.seed)
        featurizer = HBF(n_components=args.features)
    else:
        raise(NotImplementedError)

    if args.grid:
        configs = grid_configs(OPTIONS[args.dataset])
    else:
        configs = random_configs(OPTIONS[args.dataset], args.search, continuous={'alpha', 'gamma'})
    for i, config in enumerate(configs):
        if args.dataset == 'cifar':
            config['name'] = str(i)
        elif args.dataset == 'imdb':
            config['name'] = i

    if args.output:
        if not os.path.isdir('kernel/output'):
            os.mkdir('kernel/output')
        args.output = args.output[:-6] + str(int(args.output[-6:-4])+8) + args.output[-4:]
        print('Dumping output to', args.output)
        output = {}
        output['args'] = vars(args)
        output['configs'] = configs
    else:
        print('Not dumping output')
        output = None
    
    search_func = param_sweep if args.sweep else weight_sharing
    best, config, t = search_func(configs, classifier, featurizer, *load('train'), *load('val'), grad=args.grad, eta=args.eta, niter=args.niter, baseline=args.baseline, keep=args.keep, switch_to_sweep=args.switch, estimate_val=not args.exact, output=output, args=args)
    print('Config Found\t', config)
    print('Val Accuracy\t', best)
    print('Time Spent\t', t, 'seconds')

    if output is None:
        return
    output['best'] = config
    output['val'] = best

    Xtest, Ytest, Ltest = load('test')
    score = output['test'].accuracy(featurizer.transform(Xtest, **config), Ltest)
    print('Test Accuracy\t', score)
    output['test'] = score

    with open(os.path.join('kernel/output', args.output), 'wb') as f:
        pickle.dump(output, f)
    print('Dumped output to', args.output)


if __name__ == '__main__':

    main()
