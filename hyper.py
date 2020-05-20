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
import pandas as pd
import numpy as np
#from kernel.cifar import load as cifar
#from kernel.features import HashedBonGFeaturizer as HBF
#from kernel.features import RandomFourierFeaturizer as RFF
#from kernel.imdb import load as imdb
#from kernel.utils import ArrayStore, LogitClassifier, RidgeClassifier
from cifar import load as cifar
from features import HashedBonGFeaturizer as HBF
from features import RandomFourierFeaturizer as RFF
from imdb import load as imdb
from utils import ArrayStore, LogitClassifier, RidgeClassifier, SVMClassifier

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
    parser.add_argument('--svm', action='store_true', help='use SVM for IMDB')
    parser.add_argument('--search', type=int, default=32, help='size of search space')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--sweep', action='store_true', help='run parameter sweep')
    parser.add_argument('--switch', type=int, default=0, help='switch to sweep when this many configs left')
    parser.add_argument('--alpha', type=float, default=0.5, help='Ridge regularization coefficient')
    parser.add_argument('--coef', type=float, default=1.0, help='Logit regularization coefficient')
    parser.add_argument('--exact', action='store_true', help='compute exact val acc for each config')
    parser.add_argument('--grad', action='store_true', help='use exponentiated gradient instead of successive elimination')
    parser.add_argument('--sha_weights', action='store_true', help='use successive halving on the number of parameters')
    parser.add_argument('--no_randomize', action='store_true', help='for IMDB, do not randomize the hash for each config')
    parser.add_argument('--full_analysis', action='store_true', help='evaluate every config for ICML analysis')
    parser.add_argument('--eta', type=float, default=5.0, help='learning rate for exponentiated gradient')
    parser.add_argument('--niter', type=int, default=5, help='number of iterations for exponentiated gradient')
    parser.add_argument('--baseline', action='store_true', help='use baseline when computing gradient')
    parser.add_argument('--keep', type=float, default=0.5, help='proportion of configurations to keep in successive elimination')
    parser.add_argument('--growth', type=float, default=2.0, help='growth factor for weights if using SHA+WS')
    parser.add_argument('--output', type=str, help='name of output dump file')

    return parser


def param_sweep(configs, classifier, featurizer, Xtrain, Ytrain, Ltrain, Xval, Yval, Lval, array_store=None, start_time=None, output=None, Xtest=None, Ytest=None, Ltest=None, args=None, num_cfgs=32, **kwargs):
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
        Xtest: test data
        Ytest: test label encodings
        Ltest: test labels
        args: the command line arguments
        num_cfgs: number of configs to evaluate
        kwargs: ignored
    Returns:
        best validation accuracy, corresponding configuration, time spent
    '''

    if start_time is None:
        featurizer.fit(Xtrain, args=args)

    if not output is None:
        output.setdefault('time', [])
        output.setdefault('scores', [])
        best = 0.0

    df = pd.DataFrame(columns = ['round', 'dim', 'cfg_number', 
                                 'st_train_acc', 'st_val_acc', 'st_test_acc', 'st_train_time'])

    end_eval = 0
    t = time.time() if start_time is None else start_time
    scores = []
    for config in configs[:num_cfgs]:

        print('Mapping', end='\r')
        FYtrain = featurizer.fit_transform(Xtrain, y=Ytrain, **config)

        print('Fitting', end='\r')
        st_fit_start = time.time()
        clf = classifier()
        clf.fit(*FYtrain)
        st_train_time = time.time() - st_fit_start

        print('Scoring', end='\r')
        Fval = featurizer.transform(Xval, **config) if array_store is None else array_store.get(config['name'])
        score = clf.accuracy(Fval, Lval)

        ### EVAL ### time is excluded
        start_eval = time.time()

        # Get validation score
        #Fval = featurizer.transform(Xval, **config)
        #score = clf.accuracy(Fval, Lval)

        # Get test score
        Ftest = featurizer.transform(Xtest, **config)
        score_test = clf.accuracy(Ftest, Ltest)

        # Get oracle validation score
        score_oracle = score

        # Get oracle test score
        score_test_oracle = score_test

        end_eval += time.time() - start_eval
        ### end EVAL ### time is excluded

        if args.full_analysis:
            F_train = featurizer.transform(Xtrain, **config)
            st_train_acc = clf.accuracy(F_train, Ltrain)
            st_val_acc = score
            st_test_acc = score_test
            # Append this data to df
            new_row = [-1, args.features, config['name'], 
                        st_train_acc, st_val_acc, st_test_acc, st_train_time]

            print('[full_analysis] new_row:', new_row)
            df.loc[len(df)] = new_row

        scores.append(score)
        watch = time.time() - t - end_eval
        print('\tVal:', round(score, 5),
              '\tTest:', round(score_test, 5),
              '\tO-Val:', round(score_oracle, 5),
              '\tO-Test:', round(score_test_oracle, 5),
              '\tTime:', round(watch), 'seconds',
              '\tConfig:', config)
        if not output is None:
            output['time'].append(watch)
            if score >= best:
                best = score
                output['test'] = clf

        gc.collect()

    if args.full_analysis:
        df_filename = ''
        if args.dataset == 'cifar':
            df_filename = 'results/final/analyses/analysis_cifar_{}_ST.pkl'
        elif args.dataset =='imdb':
            if args.no_randomize:
                df_filename = 'results/final/analyses/analysis_imdb_norand_{}_ST.pkl'
            else:
                df_filename = 'results/final/analyses/analysis_imdb_{}_ST.pkl'

        print('[full_analysis] df_filename:', df_filename.format(args.features))
        # Save dataframe to disk
        df.to_pickle(df_filename.format(args.features))
        print('[full_analysis] saved!')

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

def weight_sharing(configs, classifier, featurizer, Xtrain, Ytrain, Ltrain, Xval, Yval, Lval, grad=False, eta=10.0, niter=5, baseline=False, keep=0.5, switch_to_sweep=0, estimate_val=True, output=None, Xtest=None, Ytest=None, Ltest=None, args=None):
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
        Xtest: test data
        Ytest: test label encodings
        Ltest: test labels
        args: the command line arguments
    Returns:
        (best validation accuracy, corresponding configuration), time spent
    '''

    featurizer.fit(Xtrain, args=args)
    dist = np.ones(len(configs)) / len(configs)
    update_dist = exponentiated_gradient if grad else successive_elimination
    niter = niter if grad else int(np.ceil(np.log(len(configs)) / np.log(1.0 / keep)))
    baseline_avg = 0.0
    
    if not output is None:
        output['time'] = []
        output['scores'] = []
        output['dist'] = [np.copy(dist)]

    df = pd.DataFrame(columns = ['round', 'dim', 'cfg_number', 
                                 'ws_train_acc', 'ws_val_acc', 'ws_test_acc', 'ws_train_time'])
    t = time.time()
    end_eval = 0
    with TemporaryFile(dir=os.path.join(os.getcwd(), 'kernel')) as tf:
        with ArrayStore(RAM, len(Xval) * featurizer.n_components, tf) as store:
            for i in range(niter):

                ### SHA on params
                
                if args.sha_weights:
                    R_max = args.features
                    eta = args.growth #1.0 / keep
                    sha_feats = int(np.ceil(R_max * (eta ** (i - niter))))

                    if args.dataset == 'cifar':
                        featurizer = RFF(n_components=sha_feats)
                    elif args.dataset == 'imdb':
                        featurizer = HBF(n_components=sha_feats, randomize=(not args.no_randomize))
                    else:
                        raise(NotImplementedError)
                    print("[sha_weights] Num features:", sha_feats)
                    featurizer.fit(Xtrain, args=args)

                ### end SHA on params

                print('Mapping', end='\r')
                FYtrain = featurizer.fit_transform(Xtrain, y=Ytrain, cfg2idx=zip(configs, random_assignment(len(Xtrain), dist)))

                print('Fitting', end='\r')
                ws_fit_start = time.time()
                clf = classifier()
                clf.fit(*FYtrain)
                ws_train_time = time.time() - ws_fit_start

                print('Scoring', end='\r')
                dist, scores, configs = update_dist(clf, featurizer, Xval, Lval, dist, configs, eta=eta, keep=keep, store=None if estimate_val else store, baseline=baseline_avg)
                if baseline:
                    baseline_avg += (dist * (1.0-np.array(scores))).sum()
                    baseline_avg /= i + 1


                #### FULL ANALYSIS ####
                if args.full_analysis and not args.sha_weights:
                    print('[full_analysis] round:', i)

                    for cfg in configs:
                        print('[full_analysis] cfg[\'name\']:', cfg['name'])

                        FY_cfg_train = featurizer.fit_transform(Xtrain, y=Ytrain, **cfg)
                        F_cfg_train = featurizer.transform(Xtrain, **cfg)
                        F_cfg_val = featurizer.transform(Xval, **cfg)
                        F_cfg_test = featurizer.transform(Xtest, **cfg)

                        # Get WS scores for each of these
                        ws_train_acc = clf.accuracy(F_cfg_train, Ltrain)
                        ws_val_acc = clf.accuracy(F_cfg_val, Lval)
                        ws_test_acc = clf.accuracy(F_cfg_test, Ltest)

                        # Append this data to df
                        new_row = [i, args.features, cfg['name'], 
                                   ws_train_acc, ws_val_acc, ws_test_acc, ws_train_time]

                        print('[full_analysis] new_row:', new_row)
                        df.loc[len(df)] = new_row

                #### END FULL ANALYSIS ####
                    
                best, config = max(zip(dist, scores, configs), key=itemgetter(0))[1:] if grad else max(zip(scores, configs), key=itemgetter(0))
                
                ### EVAL ### time is excluded
                start_eval = time.time()

                # Get validation score
                Fval = featurizer.transform(Xval, **config)
                score = clf.accuracy(Fval, Lval)

                # Get test score
                Ftest = featurizer.transform(Xtest, **config)
                score_test = clf.accuracy(Ftest, Ltest)

                # Train oracle
                if args.dataset == 'cifar':
                    oracle_fzr = RFF(n_components=args.features)
                elif args.dataset == 'imdb':
                    oracle_fzr = HBF(n_components=args.features, randomize=(not args.no_randomize))
                else:
                    raise(NotImplementedError)
                oracle_fzr.fit(Xtrain, args=args)
                FYtrain = oracle_fzr.fit_transform(Xtrain, y=Ytrain, **config)
                clf = classifier()
                clf.fit(*FYtrain)

                # Get oracle validation score
                Fval = oracle_fzr.transform(Xval, **config)
                score_oracle = clf.accuracy(Fval, Lval)

                # Get oracle test score
                Ftest = oracle_fzr.transform(Xtest, **config)
                score_test_oracle = clf.accuracy(Ftest, Ltest)

                end_eval += time.time() - start_eval
                ### end EVAL ### time is excluded

                watch = time.time() - t - end_eval
                ###print('Val:', round(best, 5),
                print('\tVal:', round(score, 5),
                      '\tTest:', round(score_test, 5),
                      '\tO-Val:', round(score_oracle, 5),
                      '\tO-Test:', round(score_test_oracle, 5),
                      '\tTime:', round(watch), 'seconds',
                      '\tConfig:', config)
                if not output is None:
                    output['time'].append(watch)
                    output['dist'].append(np.copy(dist))
                    output['scores'].append(np.array(scores))

                if sum(dist > 0.0) <= switch_to_sweep and not grad:
                    return param_sweep(configs, classifier, featurizer, Xtrain, Ytrain, Ltrain, Xval, Yval, Lval, array_store=None if estimate_val else store, start_time=t, output=output)

                gc.collect()

            if args.sha_weights:
                if args.dataset == 'cifar':
                    featurizer = RFF(n_components=args.features)
                elif args.dataset == 'imdb':
                    featurizer = HBF(n_components=args.features, randomize=(not args.no_randomize))
                else:
                    raise(NotImplementedError)
                print("[sha_weights] Num features:", args.features)
                featurizer.fit(Xtrain, args=args)

            if not niter:
                config = configs[0]

            print('Mapping', end='\r')
            FYtrain = featurizer.fit_transform(Xtrain, y=Ytrain, **config)

            print('Fitting', end='\r')
            clf = classifier()
            clf.fit(*FYtrain)

            print('Scoring', end='\r')
            Fval = featurizer.transform(Xval, **config) if estimate_val else store.get(config['name'])

            #### FULL ANALYSIS ####
            if args.full_analysis and not args.sha_weights:
                print('[full_analysis] round:', niter)

                for cfg in configs:
                    print('[full_analysis] cfg[\'name\']:', cfg['name'])

                    # Featurize train, val, test using this cfg
                    FY_cfg_train = featurizer.fit_transform(Xtrain, y=Ytrain, **cfg)
                    F_cfg_train = featurizer.transform(Xtrain, **cfg)
                    F_cfg_val = featurizer.transform(Xval, **cfg)
                    F_cfg_test = featurizer.transform(Xtest, **cfg)

                    # Get WS scores for each of these
                    ws_train_acc = clf.accuracy(F_cfg_train, Ltrain)
                    ws_val_acc = clf.accuracy(F_cfg_val, Lval)
                    ws_test_acc = clf.accuracy(F_cfg_test, Ltest)

                    # Append this data to df
                    new_row = [niter, args.features, cfg['name'], 
                                ws_train_acc, ws_val_acc, ws_test_acc, ws_train_time]

                    print('[full_analysis] new_row:', new_row)

                    df.loc[len(df)] = new_row

                df_filename = ''
                if args.dataset == 'cifar':
                    df_filename = 'results/final/analyses/analysis_cifar_{}_WS.pkl'
                elif args.dataset =='imdb':
                    if args.no_randomize:
                        df_filename = 'results/final/analyses/analysis_imdb_norand_{}_WS.pkl'
                    else:
                        df_filename = 'results/final/analyses/analysis_imdb_{}_WS.pkl'

                print('[full_analysis] df_filename:', df_filename.format(args.features))
                # Save dataframe to disk
                df.to_pickle(df_filename.format(args.features))
                print('[full_analysis] saved!')

            ### END FULL ANALYSIS


            ### EVAL ### time is excluded
            start_eval = time.time()

            # Get validation score
            Fval = featurizer.transform(Xval, **config)
            score = clf.accuracy(Fval, Lval)
            # Get test score
            Ftest = featurizer.transform(Xtest, **config)
            score_test = clf.accuracy(Ftest, Ltest)

            # Get oracle validation score
            score_oracle = score

            # Get oracle test score
            score_test_oracle = score_test

            end_eval += time.time() - start_eval
            ### end EVAL ### time is excluded
            
            watch = time.time() - t - end_eval
            print('Val:', round(score, 5),
                  '\tTest:', round(score_test, 5),
                  '\tO-Val:', round(score_oracle, 5),
                  '\tO-Test:', round(score_test_oracle, 5),
                  '\tTime:', round(watch), 'seconds',
                  '\tConfig:', config)
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
        if args.svm:
            print('[svm] using SVM classifier...')
            classifier = lambda: SVMClassifier(random_state=args.seed, C=args.coef, tol=1e-5, dual=True, loss='hinge')
        else:
            classifier = lambda: LogitClassifier(C=args.coef, solver='liblinear', random_state=args.seed)
        featurizer = HBF(n_components=args.features, randomize=(not args.no_randomize))
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

    Xtest, Ytest, Ltest = load('test')
    search_func = param_sweep if args.sweep else weight_sharing
    best, config, t = search_func(configs, classifier, featurizer, *load('train'), *load('val'), grad=args.grad, eta=args.eta, niter=args.niter, baseline=args.baseline, keep=args.keep, switch_to_sweep=args.switch, estimate_val=not args.exact, output=output, Xtest=Xtest, Ytest=Ytest, Ltest=Ltest, args=args)
    print('Config Found\t', config)
    print('Val Accuracy\t', best)
    print('Time Spent\t', t, 'seconds')

    if output is None:
        return
    output['best'] = config
    output['val'] = best

    score = output['test'].accuracy(featurizer.transform(Xtest, **config), Ltest)
    print('Test Accuracy\t', score)
    output['test'] = score

    with open(os.path.join('kernel/output', args.output), 'wb') as f:
        pickle.dump(output, f)
    print('Dumped output to', args.output)


if __name__ == '__main__':

    main()
