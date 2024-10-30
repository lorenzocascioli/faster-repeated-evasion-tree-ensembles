import numpy as np
import math
import scipy.stats as stats
import pickle
from utils import *

def acceptance_interval_old(N, M, n, alpha):
  def P_m(x):
    return 1-stats.hypergeom.cdf(x, N, M, n)
  for i in np.arange(n, -1, -1):
    p = P_m(i)
    if p >= 1-alpha:
      return i
  return 0

def acceptance_interval(N, M, n, alpha):
  return _acceptance_interval_impl(N, M, n, alpha, 0, n)

def _acceptance_interval_impl(N, M, n, alpha, low, high):
  def P_m(x):
    return 1-stats.hypergeom.cdf(x, N, M, n)
  if high-low < 10:
    for i in np.arange(high, low-1, -1):
      p = P_m(i)
      if p >= 1-alpha:
        return i
  else:
    mid = (low+high) // 2
    if P_m(mid) >= 1-alpha:
      return _acceptance_interval_impl(N, M, n, alpha, mid, high)
    else:
      return _acceptance_interval_impl(N, M, n, alpha, low, mid)
  return 0

def confidence_interval_old(x, N, n, alpha):
  for M in range(math.floor(x*N/n), N):
    bound = acceptance_interval(N, M, n, alpha)
    if bound > x:
      return M
  return N

def confidence_interval(x, N, n, alpha):
  low = math.floor(x*N/n)
  high = N
  return _confidence_interval_impl(x, N, n, alpha, low, high)

def _confidence_interval_impl(x, N, n, alpha, low, high):
  if high-low < 10:
    for M in range(low, high+1):
      if acceptance_interval(N, M, n, alpha) > x:
        return M
  else:
    mid = (low+high) // 2
    if acceptance_interval(N, mid, n, alpha) > x:
      return _confidence_interval_impl(x, N, n, alpha, low, mid)
    else:
      return _confidence_interval_impl(x, N, n, alpha, mid, high)
  return N


# Revised code for feature extraction using method implemented above

def extract_feats(counts, percentile=10):
    top_count_threshold = np.percentile(list(counts.values()), 100 - percentile)
    top_features = [feature for feature, count in counts.items() if count >= top_count_threshold]
    return top_features

def extract_counts(counts, base_examples, adv_examples):
    """ count how often each feature is modified """
    for i in range(len(base_examples)):
        base_ex = base_examples[i]
        adv_ex = adv_examples[i]
        for j in range(len(base_ex)):
            if np.abs(base_ex[j] - adv_ex[j]) > 1e-5:
                counts[j] +=1
    return counts

def get_features(d, at, indices, fold, delta, attack_type, model_type, \
                    num_examples, N, fnr, guard, table='result'):
    import tqdm
    from attacks import do_veritas, do_kantchelian

    assert len(indices)>=N, f"Not enough examples to work with feats_n={N}"

    time_feats = 0 #counter

    # INIT: extract full adv. ex. and divide them in possible rounds
    full_advs = TABLES[table].select()\
                .where(
                        (TABLES[table].fold==fold) &\
                        (TABLES[table].delta==delta) &\
                        (TABLES[table].mode==Attack.Mode.Full)&\
                        (TABLES[table].result!=Attack.Result.MISCLASSIFIED)\
                    )\
                .dicts().execute()

    # only take the first N full searches here, won't use more than that
    full_advs = full_advs[:N]
    # make this a dict with key=index
    full_runs = {}
    for i in full_advs:     full_runs[i['index']]=i 

    # divide the indices of all our N full runs into the different rounds
    n = int(N/FS_ROUNDS)
    ns = {}
    for i in range(FS_ROUNDS):
        ns[i] = list(full_runs.keys())[i*n:(i+1)*n]
    # no run time counted from stuff above: related to loading full searches

    to_insert = [] # will insert in db runs done here as both pruned and mixed

    # ROUND 1: full-only to inizialize counts
    t = time.time()
    base_examples, adv_examples = [], []
    round_advs = [i for i in full_runs.values() if i['index'] in ns[0]]
    for run in round_advs:

        # make a copy for pruned/mixed (on these, they also do full directly)
        mixed_res, pruned_res = run.copy(), run.copy()
        pruned_res['mode'] = Attack.Mode.Pruned
        mixed_res['mode'] = Attack.Mode.Mixed
        to_insert.extend([pruned_res, mixed_res])

        if run['result']==str(Attack.Result.SAT):
            base_examples.append(pickle.loads(run['base_example']))
            adv_examples.append(pickle.loads(run['adv_example']))
    full_only = ns[0].copy()
    counts = dict.fromkeys(range(adv_examples[0].shape[0]), 0)
    counts = extract_counts(counts, base_examples, adv_examples)

    # build threhsold on false negative rate for statistical test
    # (!) upper bound on #FN from a sample of size n < this value
    #     --> we're 90% sure it's #FN < FNR on the full population N
    fn_bound = int(fnr*N)

    # ROUNDS 2-end: mixed to grow feature set - stop if test ok
    mode = Attack.Mode.Pruned
    max_time = get_timeout(d, attack_type, model_type, mode)
    percentile = FS_START*100
    fs = extract_feats(counts, percentile)
    print(f"starting pruned search with {len(fs)} feats.")
    # to_insert = [] # will insert in db runs done here as both pruned and mixed
    done = []

    # add time for: round 1 + statistical test init. + rounds 2-end init.
    time_feats += time.time() - t 

    for round_ in range(1, FS_ROUNDS):
        round_runs = []
        # fnr_round = 0
        fn_round = 0
        done.extend(ns[round_].copy())

        for i in ns[round_]:
            # do pruned search
            if attack_type == Attack.Type.Veritas:
                res = do_veritas(d, i, at, fold, delta, mode, fs, max_time, guard)
            elif attack_type == Attack.Type.Kantchelian:
                res = do_kantchelian(d, i, at, fold, delta, mode, fs, max_time, guard)
            else:
                print("ERROR - unknown attack type.")

            if res['result']!=Attack.Result.SAT:
                # we run full and add pruned time to the res
                pruned_time = res['time']
                pruned_result = res['result']
                res = full_runs[i]
                res['time'] += pruned_time

                if  pruned_result==Attack.Result.UNSAT and \
                        res['result']==str(Attack.Result.SAT):
                    # false neg
                    # fnr_round += 1/n
                    fn_round += 1
            round_runs.append(res)

        # update counts with new adv. ex. generated in this round
        # (now considering both the pruned and the full ones)
        t = time.time()
        base_examples, adv_examples = [], []
        for run in round_runs:
            if run['result']==Attack.Result.SAT:
                base_examples.append(pickle.loads(run['base_example']))
                adv_examples.append(pickle.loads(run['adv_example']))
        counts = extract_counts(counts, base_examples, adv_examples)
        time_feats += time.time() - t # add time to update counts

        # manually create two runs for each one, for pruned/mixed search
        for res in round_runs:
            pruned_res, mixed_res = res.copy(), res.copy()
            pruned_res['mode'] = Attack.Mode.Pruned
            mixed_res['mode'] = Attack.Mode.Mixed
            to_insert.extend([pruned_res, mixed_res])

        # execute test to see if fs is enough or there are too many false negs.
        t = time.time()
        # if fnr_round > fnr-Delta:
        ub_fn = confidence_interval(fn_round, N, n, 0.1)
        if ub_fn > fn_bound:
            # FNR too high: double features (5-->10-20-30-40)
            if percentile == 5:     percentile *= 2
            else:       percentile += 10
            # percentile *= 2
            fs = extract_feats(counts, percentile)
            # print(f"HA! Had {fnr_round} False Negs vs accept threshold {fnr-Delta} --> new percentile: {percentile}")
            print(f"HA! Had {fn_round} False Negs --> upper bound {ub_fn} vs accept threshold {fn_bound} --> new percentile: {percentile}")
            print(f"now using {len(fs)} feats.")
            time_feats += time.time() - t # add statistical test time
        else:
            # print(f"no need for new feats here (fnr: {fnr_round}): done!")
            print(f"no need for new feats here (fn: {fn_round}--> upper bound {ub_fn}, bound: {fn_bound}): done!")
            time_feats += time.time() - t 
            break

    # remove all indices already processed during feature selection 
    # = all those from rounds 1-->end
    i_exclude = full_only + done
    indices = [i for i in indices if i not in i_exclude]

    bulk_insert(to_insert, fold)
    print(f"results from feature selection rounds 2-end inserted in db.")

    print(f"tot selected feats: {len(fs)}, added time (except runs): {time_feats}")

    return fs, indices, time_feats, i_exclude #full_only






