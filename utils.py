import time
import os
import json
import joblib
import math
import numpy as np
#import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import groot.model
import prada, veritas
from veritas import FloatT
from db import *


##################             1. Constants                 ##################

SEED = 6 
NFOLDS = 5
GUARD = 1e-5 # max possible numerical guard
RUN_TIMEOUT = 6*60*60 # 6h
FNR = 0.25 # tolerated false negative rate in pruned 
FS_START = 0.05 # start selecting 5% feats, and if needed increase
FS_ROUNDS = 5 # rounds in the feature selection procedure
MODEL_DIR = os.environ["DATA_AND_TREES_MODEL_DIR"]

#########################     2. Model Stuff       #########################


def get_params_xgb(d, fold):
    params = {
        "random_state": d.seed+17*fold,
        "n_jobs": 1,
        # "n_estimators": [100],
        # "max_depth": [6],
        # "learning_rate": [0.1],
        # "subsample": [0.25],
        # "colsample_bytree": 0.8,
        # "tree_method": "hist",
        "objective": "binary:logistic",
        "eval_metric": "error",
    }
    if d.is_multiclass():
        raise RuntimeError(f"multi-class problem?!")
    return params

def get_params_rf(d, fold):
    params = {
        "random_state": d.seed+17*fold,
        "n_jobs": 1,
        # "n_estimators": 100,
        # "max_features": None,
        # "max_depth": None,
        "max_leaf_nodes": 254
    }
    return params

def get_params_groot(d, fold):
    params = {
        "random_state": d.seed+17*fold,
        "n_jobs":1, 
        "min_samples_leaf": 2
    }
    return params

def get_params(d, model_type, fold):
    if model_type == "rf":
        return get_params_rf(d, fold)
    elif model_type == "xgb":
        return get_params_xgb(d, fold)
    elif model_type == "groot":
        return get_params_groot(d, fold)
    else:
        raise RuntimeError(f"get_params: model type {model_type}")

# def num_params(d, model_type):
#     param_dict = get_params(d, model_type)
#     return sum(1 for params in d.paramgrid(**param_dict))

def get_model_name(d, fold, model_type, num_trees, tree_depth, **kwargs):
    a = "-".join(f"{k}{v}" for k, v in kwargs.items())
    return f"{d.name()}-seed{d.seed}-fold{fold}_{num_trees}-{tree_depth}{a}.{model_type}"


def get_model(d, fold, model_type, params, attack_type, cache=True):
    if model_type == "xgb":
        model, meta = get_xgb(d, fold, params["xgb"], cache)
    elif model_type == "rf":
        model, meta = get_rf(d, fold, params["rf"], attack_type, cache)
    elif model_type == "groot":
        model, meta = get_groot(d, fold, params["groot"], attack_type, cache) 
    else:
        # TODO: add lightgbm/groot?
        raise RuntimeError(f"Not ready for {model_type} yet!")

    if not d.is_binary():
        raise RuntimeError(f"This is not binary clf!")

    # model loaded, now convert to addtree
    at = veritas.get_addtree(model)
    
    # test against floating issues
    xtrain, ytrain, xtest, ytest = d.train_and_test_set(fold_index_or_fraction=fold, nfolds=NFOLDS)
    is_correct = veritas.test_conversion(at, xtest, model.predict_proba(xtest)[:,1])
    assert is_correct==True, f"Something wrong with conversion to addtree!"

    # old check
    # yhat_test = model.predict(xtest)
    # yat_test = (at.eval(xtest)>0).ravel()
    # at_error = np.mean(np.abs(yat_test - yhat_test))
    # #print(f"{np.sum(np.abs(yat_test - yhat_test))}/{len(ytest)}")
    # #print(at_error)
    # if at_error>GUARD:
    #     print(f"WARNING: Conversion to addtree modified accuracy (at_error={at_error})! Check!")

    print(f"{model_type} model (acc.: {meta['test_accuracy']})")
    return (model, meta, at)


def get_xgb(d, fold, params, cache):
    # if d.task == prada.Task.REGRESSION:
    #     raise RuntimeError("nope")
    
    # load default dataset params from prada
    # additional_params = d.xgb_params(d.task)
    additional_params = get_params(d, 'xgb', fold) 
    for k, v in additional_params.items():
        params[k] = v

    # params["random_state"] = d.seed+17*fold
    # params["nthread"] = 1 

    # model_name = d.get_model_name(fold, "xgb", params["n_estimators"], params["max_depth"], 
    #                                 lr=f"{params['learning_rate']*100:.0f}")
    model_name = get_model_name(d, fold, "xgb", params["n_estimators"], params["max_depth"], 
                                    lr=f"{params['learning_rate']*100:.0f}")
    model_path = os.path.join(MODEL_DIR, model_name)

    if os.path.isfile(model_path) and cache:
        print(f"loading XGB model from file: {model_path}")
        bst, meta = joblib.load(model_path)
    else: 
        # train model
        xtrain, ytrain, xtest, ytest = d.train_and_test_set(fold_index_or_fraction=fold, nfolds=NFOLDS)

        t = time.time()
        bst = XGBClassifier(**params)
        bst.fit(xtrain, ytrain)
        t = time.time() - t
        print(f'new xgb ensemble trained in {t} seconds.')

        yhat_train = bst.predict(xtrain)
        yhat_test = bst.predict(xtest)
        print(f"xgb: train acc. {acc(ytrain, yhat_train):.3f}, test acc. {acc(ytest,  yhat_test):.3f}")

        meta = {
                "params": params,
                "columns": d.X.columns,
                # "task": d.task,
                "train_accuracy": acc(ytrain, yhat_train),
                "test_accuracy": acc(ytest, yhat_test),
                "training_time": t,
        }

        if cache:   joblib.dump((bst, meta), model_path)

    return (bst, meta)


def get_rf(d, fold, params, attack_type, cache):
    # if d.task == prada.Task.REGRESSION:
    #     raise RuntimeError("nope")

    # (!) hack: need small RFs for kantchelian, or verification is veeery slow
    if attack_type==Attack.Type.Kantchelian:
        params['n_estimators']=25
        params['max_depth']=7 #8 #10

    # load default dataset params from data_and_trees
    # additional_params = d.rf_params()
    additional_params = get_params(d, 'rf', fold) 
    for k, v in additional_params.items():
        params[k] = v

    #params["random_state"] = d.seed+17*fold
    #params["n_jobs"] = 1 #temp

    #model_name = d.get_model_name(fold, "rf", params["n_estimators"], params["max_depth"])
    model_name = get_model_name(d, fold, "rf", params["n_estimators"], params["max_depth"])
    model_path = os.path.join(MODEL_DIR, model_name)

    if os.path.isfile(model_path) and cache:
        print(f"loading RF model from file: {model_path}")
        rf, meta = joblib.load(model_path)
    else: 

        # train model
        xtrain, ytrain, xtest, ytest = d.train_and_test_set(fold_index_or_fraction=fold, nfolds=NFOLDS)

        t = time.time()
        rf = RandomForestClassifier(**params)
        rf.fit(xtrain, ytrain)
        t = time.time() - t
        print(f'new rf ensemble trained in {t} seconds.')

        yhat_train = rf.predict(xtrain)
        yhat_test = rf.predict(xtest)
        print(f"rf: train acc. {acc(ytrain, yhat_train):.3f}, test acc. {acc(ytest,  yhat_test):.3f}")

        meta = {
                "params": params,
                "columns": d.X.columns,
                # "task": d.task,
                "train_accuracy": acc(ytrain, yhat_train),
                "test_accuracy": acc(ytest, yhat_test),
                "training_time": t,
        }

        if cache:   joblib.dump((rf, meta), model_path)

    return (rf, meta)


def get_groot(d, fold, params, attack_type, cache):
    # if d.task == prada.Task.REGRESSION:
    #     raise RuntimeError("nope")

    # (!) hack: need small GROOT forests for kantchelian, or verification is veeery slow
    if attack_type==Attack.Type.Kantchelian:
        params['n_estimators']=25
        params['max_depth']=7 #8 #10

    # load default dataset params from data_and_trees
    additional_params = get_params(d, 'groot', fold)
    for k, v in additional_params.items():
        params[k] = v

    # params["random_state"] = d.seed+17*fold

    model_name = get_model_name(d, fold, "groot", params["n_estimators"], params["max_depth"],\
                                epsilon=f"{params['epsilon']*100:.0f}")
    model_path = os.path.join(MODEL_DIR, model_name)

    if os.path.isfile(model_path) and cache:
        print(f"loading GROOT model from file: {model_path}")
        grf, meta = joblib.load(model_path)
    else: 

        # train model
        xtrain, ytrain, xtest, ytest = d.train_and_test_set(fold_index_or_fraction=fold, nfolds=NFOLDS)

        params["attack_model"] = np.ones(xtrain.shape[1]) * params['epsilon']
        # remove "epsilon" from params
        params.pop("epsilon")

        t = time.time()
        grf = groot.model.GrootRandomForestClassifier(**params)
        grf.fit(xtrain, ytrain)
        t = time.time() - t
        print(f'new groot rf trained in {t} seconds.')

        yhat_train = grf.predict(xtrain)
        yhat_test = grf.predict(xtest)
        print(f"groot: train acc. {acc(ytrain, yhat_train):.3f}, test acc. {acc(ytest,  yhat_test):.3f}")

        meta = {
                "params": params,
                "columns": d.X.columns,
                #"task": d.task,
                "train_accuracy": acc(ytrain, yhat_train),
                "test_accuracy": acc(ytest, yhat_test),
                "training_time": t,
        }

        if cache:   joblib.dump((grf, meta), model_path)

    return (grf, meta)



########################     2. Other Useful Stuff       #######################

def get_timeout(d, attack, model, mode):
    """ max time: 60 sec. for full search, 1/0.1 (kan./ver.) for pruned search"""
    if mode==Attack.Mode.Full:  t=60; return t
    elif attack==Attack.Type.Kantchelian:     t=1
    elif attack==Attack.Type.Veritas:     t=0.1

    #if d.name()=="Higgs":   t = t*10 # + higgs needs longer timeout
    #if model=='rf':     t *= 10 # RFs are typically deeper, so tougher

    return t


# def exclude_misclassified_indices(d, at, indices, fold, num_examples, delta, cached):
#     # exclude (and add to db) misclassified examples
#     ok = []
#     to_insert = []
#     for j in range(len(indices)):
#         i = indices[j]
#         example, label = d.X.iloc[i, :].to_numpy(), int(d.y[i])
#         if (at.eval(example) > 0.0) != label:

#             for mode in Attack.Mode:
#                 if i in cached[mode]: continue

#                 # unseen misclassified example: add entry in db (mode-specific)
#                 res = {
#                         "index": i, 
#                         "fold": fold, 
#                         "delta": delta,
#                         "mode": mode,
#                         "hostname": os.uname().nodename,
#                         "base_label": label,
#                         "result": Attack.Result.MISCLASSIFIED,
#                         "time": 0.0
#                 }
#                 #update_or_insert_res(res)
#                 to_insert.append(res)
#         else:
#             ok.append(i)

#     bulk_insert(to_insert, fold)
#     return ok


# def get_N_test_example_indices(d, fold, N):
#     xtrain, ytrain, xtest, ytest = d.train_and_test_set(fold)
#     indices = np.copy(xtest.index)

#     # shuffle indices, take first N
#     gen = np.random.default_rng(22*fold + 6*d.nfolds + d.seed)
#     gen.shuffle(indices)

#     if d.name()=="Higgs":   N = N/10 # + higgs too slow: we use less ex.

#     return indices[:N]


def get_N_correct_test_example_indices(d, at, fold, N=None):
    xtrain, ytrain, xtest, ytest = d.train_and_test_set(fold_index_or_fraction=fold, nfolds=NFOLDS)
    # ytest_pred = (at.eval(xtest)[0] > 0.0)
    ytest_pred = [float(i[0]) for i in at.eval(xtest) > 0.0]
    test_mask = ytest==ytest_pred
    indices = np.copy(xtest.index[test_mask])

    # shuffle indices, take first N
    gen = np.random.default_rng(22*fold + 6*NFOLDS + d.seed)
    gen.shuffle(indices)

    # if d.name()=="Higgs":   N = N//10 # + higgs too slow: we use less ex.

    # return indices[:N]
    if N:   return indices[:N]
    return indices

# (!) added for hardening experiment
def get_N_correct_train_example_indices(d, at, fold, N=None):
    xtrain, ytrain, xtest, ytest = d.train_and_test_set(fold_index_or_fraction=fold, nfolds=NFOLDS)
    ytrain_pred = [float(i[0]) for i in at.eval(xtrain) > 0.0]
    train_mask = ytrain==ytrain_pred
    indices = np.copy(xtrain.index[train_mask])
    # shuffle indices, take first N
    gen = np.random.default_rng(22*fold + 6*NFOLDS + d.seed)
    gen.shuffle(indices)
    if N:   return indices[:N]
    return indices

def count_avg_model_size(at):
    """ test: count avg num. leaves in addtree """
    leaves_n = []
    for t in at:
        leaves_n.append(len(t.get_leaf_ids()))

    return np.mean(leaves_n)

def count_rel_feats_splits(at, fs):
    """ test: count how many splits in full at are on relevant feats """
    rel_splits = 0
    all_splits = at.get_splits()
    tot_splits = np.sum([len(i) for i in all_splits.values()])

    for f in fs:
        if f in all_splits.keys():
            rel_splits += len(all_splits[f])

    return rel_splits/tot_splits


def extract_guard(at):
    """ check if we need guard smaller than GUARD """
    min_diff = 1
    for attribute, split_values in at.get_splits().items():
        #print("\n", attribute, split_values)
        if len(split_values)>1:
            diffs = np.diff(split_values)
            #print(diffs)
            if np.min(diffs) < min_diff:
                #print("*** HERE ***")
                min_diff = np.min(diffs)
    
    assert min_diff > 0
    guard = 10**(int(math.floor(math.log10(min_diff))))
    guard = min(guard, GUARD)
    print(f"Running attacks with guard {guard}.\n")
    return guard


def debug_index(d, at, debug):
    print("\nDEBUGGING INDEX", debug)
    indices = np.array([int(debug)])
    example = d.X.iloc[int(debug), :]
    print("\nExample: \n", example)
    print("at.eval(example): ", at.eval(example)[0], "with label", d.y[int(debug)], "\n")
    # splits = at.get_splits()
    # keys = sorted(splits.keys())
    # for k in keys:
    #     vs = splits[k]
    #     x = example.iloc[k]
    #     m = np.argmin(np.array(vs)-x)
    #     #warning_string = ", difference < GUARD!!!" if np.abs(x-vs[m])<GUARD else " " 
    #     #min_diff = np.abs(x-vs[m])
    #     #if min_diff < GUARD:
    #     print(" -", k, x, vs[m])#, min_diff, min_diff<GUARD)

    return indices


def json_dump(fname, data):
    with open(fname,'w') as json_file:
        json.dump(data, json_file)#, cls=NpEncoder)
    print(f"\nResults written to {fname}")

def json_load(fname):
    with open(fname,'r') as json_file:
        #print(f"\nReading {fname}")
        return json.load(json_file)

def dump(fname, data):
    joblib.dump(data, fname, compress=True)
    print(f"\nResults written to {fname}")

def load(fname):
    print(f"\nReading {fname}")
    return joblib.load(fname)

def acc(ytrue, ypred):
    return accuracy_score(ytrue, ypred)

def linf(ex1, ex2):
    return np.max(np.abs(ex1 - ex2))

def l0(ex1, ex2):
    return np.sum([i>0.0 for i in np.abs(ex1 - ex2)])

def format_val_std(value, std):
    #return f"{value} \u00B1 {std}"
    return f"{value} $\\pm$ {std}"
    #return f"{value} +- {std}"

def format_val_std_2(value, std):
    return f"{value} & $\\pm$ & {std}"

def configure_matplotlib():
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",

        "legend.frameon": False,
        "legend.fancybox": False,
        #"font.size": 6,
        #"axes.linewidth": 0.5,
        #"xtick.major.width": 0.5,
        #"ytick.major.width": 0.5,
        #"xtick.minor.width": 0.5,
        #"ytick.minor.width": 0.5,
        #"lines.linewidth": 0.6,

        "svg.fonttype": "none",

        "font.size": 7,
        "axes.linewidth":    0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "lines.linewidth":   0.8,

        "hatch.linewidth": 0.5,

        #"text.latex.unicode" : False,
    })
    plt.rc("text.latex", preamble=r"\\usepackage{amsmath}")



