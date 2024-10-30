import pickle
import tqdm
import gurobipy as gu
from veritas import FloatT, Interval, KantchelianAttack, Config, HeuristicType
from feature_extraction import *
from utils import *


def get_adversarial_examples(d, at, indices, fold, delta, attack_type, \
                                model_type, mode, num_examples, feats_n, fnr, \
                                guard, read_cached, cached, debug=None):
    
    # 1. if pruned, extract relevant features
    if mode == Attack.Mode.Pruned:
        feature_set, indices, feats_time, feats_ex = \
            get_features(d, at, indices, fold, delta, attack_type, model_type, 
                            num_examples, feats_n, fnr, guard)
        meta = save_metadata(delta, fold, num_examples, feats_n, fnr, \
                                feats_time, feats_ex, feature_set, d.X.shape[1], \
                                d.name(), attack_type, model_type)
    elif mode==Attack.Mode.Mixed:
        # new ad-hoc procedure for mixed search
        flag = do_mixed_search(d, at, indices, fold, delta, attack_type, \
                                model_type,mode, range(d.X.shape[1]), guard)
        return flag

    else:   feature_set = range(d.X.shape[1]) # full

    # 2. run search
    flag = do_search(d, at, indices, fold, delta, attack_type, model_type, mode,\
                        feature_set, guard, read_cached, cached, debug)

    if not flag:    raise RuntimeError(f"Something went wrong in {mode} search.")
    return


def do_search(d, at, indices, fold, delta, attack_type, model_type, mode, \
                feature_set, guard, read_cached, cached, debug=None):

    to_insert = []
    # max_time = get_timeout(d, attack_type, mode)

    # tests to check dimensions of full vs delta-pruned vs pruned addtrees
    # log = {
    #         "num_leaves": { "full": [], "delta-pruned": [], "pruned": [] },
    #         "fs_splits": {"full": [], "delta-pruned": [], "pruned": []}
    # }
    # # evaluate model sizes for full model (all on pruned, so we know fs!)
    # if mode==Attack.Mode.Pruned:
    #     log['num_leaves']['full'].append(count_avg_model_size(at))
    #     log['fs_splits']['full'].append(count_rel_feats_splits(at, feature_set))
    log = None

    tot_time, shut_down = 0, False

    for i in tqdm.tqdm(indices):

        if read_cached and i in cached[mode].keys(): 
            # we have cached record: no need to run attack
            continue

        if shut_down:
            res = build_skipped_res(d, i, fold, delta, mode)
        else:
            # do the actual search
            # if attack_type == Attack.Type.Veritas:
            #     res = do_veritas(d, i, at, fold, delta, mode, feature_set, max_time, guard, log)
            # elif attack_type == Attack.Type.Kantchelian:
            #     res = do_kantchelian(d, i, at, fold, delta, mode, feature_set, max_time, guard, log)
            # else:
            #     print("ERROR - unknown attack type.")
            res = do_attack(d, i, at, fold, delta, attack_type, model_type, mode, \
                                feature_set, guard, log)

            if debug is not None:   
                print('\n', i, f"{len(feature_set)} considered feats", '\n', res)

            tot_time += res['time']
            if tot_time >= RUN_TIMEOUT:
                print(f"TIMEOUT EXCEEDED: shutting down {mode} search.")
                shut_down = True

        to_insert.append(res)

    print(f"{mode} search done.\n")
    #print(f"Average no. of leaves in pruned model in {mode}: {np.mean(at_sizes)}")
    #if mode==Attack.Mode.Pruned:
    #    print(f"Average ratio of splits on fs in delta-pruned addtree: {np.mean(fs_ratios)}")

    bulk_insert(to_insert, fold)
    print(f"results inserted in db.")

    # if mode==Attack.Mode.Pruned:
    #     for k1, v1 in log.items():
    #         for k2, v2 in v1.items():
    #             log[k1][k2] = np.mean(v2)
    #     print("logs on models dimensions: \n", log)
    return True

def do_attack(d, i, at, fold, delta, attack_type, model_type, mode, feats, guard, log=None):
    max_time = get_timeout(d, attack_type, model_type, mode)
    if attack_type == Attack.Type.Veritas:
        res = do_veritas(d, i, at, fold, delta, mode, feats, max_time, guard, log)
    elif attack_type == Attack.Type.Kantchelian:
        res = do_kantchelian(d, i, at, fold, delta, mode, feats, max_time, guard, log)
    else:
        print("ERROR - unknown attack type.")

    return res


def build_skipped_res(d, i, fold, delta, mode):
    base_example = d.X.iloc[i, :].astype(FloatT)
    label = d.y[i]

    res = {
            "index": i, 
            "fold": fold, 
            "delta": delta,
            "mode": mode,
            "hostname": os.uname().nodename,
            "base_label": label,
            "base_example": pickle.dumps(base_example.to_numpy()),
            "time": 0.0,
            "result": Attack.Result.SKIPPED
            }
    return res

def do_mixed_search(d, at, indices, fold, delta, attack_type, model_type, mode, feature_set, guard):

    # read metadata
    meta = read_metadata(fold, delta)
    fs = meta['fs']

    # load all runs for full and pruned
    full, pruned = read_fold_res(fold, delta)

    tot_time, shut_down = 0, False
    to_insert = []

    for i in tqdm.tqdm(indices):

        if shut_down:
            mixed_res = build_skipped_res(d, i, fold, delta, mode)
        else:

            # first feats_n examples: full only
            if i in meta['feats_ex']:
                mixed_res = full[i].copy()
                mixed_res['mode'] = Attack.Mode.Mixed
                tot_time += mixed_res['time']
                to_insert.append(mixed_res)
                continue
            
            # load/run pruned
            if i in pruned.keys() and pruned[i]['result'] not in ['MISCLASSIFIED', 'SKIP']:
                res = pruned[i].copy()
            else:
                res = do_attack(d, i, at, fold, delta, attack_type, model_type, \
                                    Attack.Mode.Pruned, fs, guard)

            if res['result']==str(Attack.Result.SAT):
                # pruned SAT: no need to run full
                mixed_res = res
            else:
                # load/run full and add pruned time
                ptime = res['time']
                if i in full.keys() and full[i]['result'] not in ['MISCLASSIFIED', 'SKIP']:
                    res = full[i]
                else:
                    res = do_attack(d, i, at, fold, delta, attack_type, model_type,\
                                        Attack.Mode.Full, feature_set, guard)
                res['time'] += ptime
                mixed_res = res

            tot_time += mixed_res['time']
            if tot_time >= RUN_TIMEOUT:
                print(f"TIMEOUT EXCEEDED: shutting down {mode} search.")
                shut_down = True
            mixed_res['mode'] = Attack.Mode.Mixed
        to_insert.append(mixed_res)

    print(f"{mode} search done.\n")

    bulk_insert(to_insert, fold)
    print(f"results inserted in db.")

    return True


def do_kantchelian(d, i, at, fold, delta, mode, feature_set, max_time, guard, log=None):

    base_example = d.X.iloc[i, :].to_numpy() #.astype(FloatT)
    label = d.y[i]

    # run attack
    res = {
            "index": i, 
            "fold": fold, 
            "delta": delta,
            "mode": mode,
            "hostname": os.uname().nodename,
            "base_label": label,
            "base_example": pickle.dumps(base_example) #.to_numpy())
    }

    tstart = time.time()

    # TODO: do we prune the addtree in addition to the search object?
    box, __, atp = prune_addtree(at, base_example, label, delta, \
                                    guard, feature_set, 'kantchelian', log)
    kan = get_kantchelian_attack(atp, base_example, label, max_time, \
                                    guard, box)
    kan.optimize()

    #res["time"] = kan.total_time
    res['time'] = time.time()-tstart

    if kan.force_stop:  
        res["result"]=Attack.Result.UNKNOWN

    elif not kan.has_solution():
        res["result"]=Attack.Result.UNSAT

    else:  
        adv_example = kan.solution()[0]
        #print(kan.solution())

        if not verify_search_outcome(res, at, base_example, label, adv_example, guard):
            res["result"]=Attack.Result.UNKNOWN
        else:
            res["result"] = Attack.Result.SAT
            res["distance"] = linf(base_example, adv_example)
            output = at.eval(adv_example)[0][0]
            res["model_output"] = -output if label else output
            res["adv_example"] = pickle.dumps(adv_example) #.to_numpy())

    return res


def do_veritas(d, i, at, fold, delta, mode, feature_set, max_time, guard, log=None):
    base_example = d.X.iloc[i, :].to_numpy() #.astype(FloatT)
    label = d.y[i]

    # run attack
    res = {
            "index": i, 
            "fold": fold, 
            "delta": delta,
            "mode": mode,
            "hostname": os.uname().nodename,
            "base_label": label,
            "base_example": pickle.dumps(base_example) #.to_numpy())
    }

    tstart = time.time()

    # TODO: do we prune the addtree in addition to the search object?
    box, at_, atp = prune_addtree(at, base_example, label, delta,\
                                    guard, feature_set, 'veritas', log)
    ver = get_veritas_search(atp, base_example, label, max_time, guard, box)

    try:
        stop_reason = ver.step_for(max_time, 100)
    except RuntimeError as e:
        print("Out of memory")
        print(e)

    #res["time"] = ver.time_since_start()
    res['time'] = time.time()-tstart

    # collect results
    if ver.num_solutions() > 0:
        res["result"] = Attack.Result.SAT
        sol = ver.get_solution(0)
        adv_example = veritas.get_closest_example(sol, base_example, guard)

        if not verify_search_outcome(res, at, base_example, label, adv_example, guard):
            res["result"]=Attack.Result.UNKNOWN
        else:

            res["distance"] = linf(base_example, adv_example)
            # assert res["distance"] > guard
            res["model_output"] = at_.eval(adv_example)[0][0]
            res["adv_example"] = pickle.dumps(adv_example) #.to_numpy())

    elif stop_reason in [  veritas.StopReason.NO_MORE_OPEN,\
                            veritas.StopReason.OPTIMAL]:
        res["result"] = Attack.Result.UNSAT
        
    elif stop_reason in [   veritas.StopReason.NONE, \
                            veritas.StopReason.OUT_OF_TIME]:
        res["result"] = Attack.Result.UNKNOWN 

    else:
        raise RuntimeError("Unknown veritas.StopReason!")

    return res


def prune_addtree(at, ex, label, delta, guard, feature_set, attack, log=None):
    if attack=='veritas':
        at_ = at.negate_leaf_values() if label else at
    elif attack=='kantchelian':
        at_ = at

    if log:
        box_ = [Interval(ex[f]-delta, ex[f]+delta) for f in range(len(ex))]
        atp_ = at_.prune(box_)
        log['fs_splits']['delta-pruned'].append(count_rel_feats_splits(atp_, feature_set))
        log['num_leaves']['delta-pruned'].append(count_avg_model_size(atp_))

    box = [Interval(ex[f]-delta, ex[f]+delta) if f in feature_set \
           else Interval.constant(ex[f]) for f in range(len(ex)) ]

    atp = at_.prune(box)

    if log:
        log['fs_splits']['pruned'].append(count_rel_feats_splits(atp, feature_set))
        log['num_leaves']['pruned'].append(count_avg_model_size(atp))

    return box, at_, atp

def get_kantchelian_attack(at, ex, label, max_time, guard, box):
    target_output = int(not bool(label))
    kan = KantchelianAttack(at, target_output=target_output, example=ex, \
                            max_time=max_time, silent=True, guard=guard)
    kan.constrain_to_box(box)
    return kan

def get_veritas_search(at, ex, label, max_time, guard, box):
    config = Config(HeuristicType.MAX_OUTPUT)
    config.stop_when_optimal = True # not needed?
    config.stop_when_num_solutions_exceeds = 1
    config.ignore_state_when_worse_than = 0.0
    config.focal_eps = 0.1 #0.05 #0.2 
    config.max_focal_size = 1000
    config.max_memory = 16*1024*1024*1024 # precaution

    search = config.get_search(at, box)
    return search

def verify_search_outcome(res, at, base_example, base_label, adv_example, guard): 
    # 1. make sure that adv_ex is indeed wrongly classified
    expected_y = int(not bool(base_label))
    y_score = at.eval(adv_example)[0][0]
    y_at = int(y_score > 0.0)
    # if y_score == 0.0: # tie breaker
    if np.isclose(y_score, 0.0, atol=guard): # tie breaker
        print("TIE BREAKER!! y_score=0.0")
        y_at = expected_y
        
    # assert y_at == expected_y, \
    #     f"{print_key(res)} " \
    #     f"- label of generated adv. ex. doesn't match!:" \
    #     f" y_at={y_at} (score={y_score:.7f}), expected_y {expected_y}" 

    # softer check
    if not y_at == expected_y:
        warning = f"{print_key(res)} " \
        f"- label of generated adv. ex. doesn't match!:" \
        f" y_at={y_at} (score={y_score:.7f}), expected_y {expected_y}" 
        print(warning)
        return False
    else:
        return True

