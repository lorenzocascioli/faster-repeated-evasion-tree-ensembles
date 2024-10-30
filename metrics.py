import pickle
import numpy as np
import pandas as pd
from db import *
#from tests import *
from utils import *

RES_COLUMNS =  ['#sat', '#unsat', '#unk', '#miscl', '#skipped', \
                    'avg_sat_dist', 'avg_sat_output',\
                    #'adv_acc', 
                    'time', 
                    # 'avg_time', 'stdev_time', # tmp
                    'false_negs', 'full_calls']

# NOTE this completely changes if db structure changes, related to db.py!

def gather_results(delta, fold, at=None, table='result'):

    # 1. read full+pruned+mixed for each index
    fold_condition = (TABLES[table].fold==fold) 
    delta_condition = (TABLES[table].delta==delta)

    read_all = TABLES[table]\
                .select(
                        TABLES[table].index,
                        TABLES[table].base_example, #test purposes
                        TABLES[table].base_label,
                        TABLES[table].mode,
                        TABLES[table].time,
                        TABLES[table].result,
                        TABLES[table].adv_example, #test purposes
                        TABLES[table].distance,
                        TABLES[table].model_output
                        )\
                .where(fold_condition & delta_condition)\
                .namedtuples().execute()
                # (!) added order_by post-ICML to tackle weird ordering issue
                # .order_by(TABLES[table].index)\

    pruned, mixed, full = [], [], []
    for (i, ex, label, mode, time, res, adv_ex, dist, out) in read_all:

        #nope: done before now
        #if out is not None:     out = -out if label else out 

        row = {
                "index": i,
                "base_example": pickle.loads(ex) if ex else None,
                "base_label": label,
                "mode": mode,
                "time": time,
                "result": res,
                "adv_example": pickle.loads(adv_ex) if adv_ex else None,
                "distance": dist, 
                "output": out
            }
        if mode==str(Attack.Mode.Pruned):    
            pruned.append(row)
        elif mode==str(Attack.Mode.Mixed):
            mixed.append(row)
        elif mode==str(Attack.Mode.Full):
            full.append(row)

    if len(full)==0 and len(pruned)==0:
        print(f"no results for fold {fold}.")
        return None

    # 1. full search
    df = pd.DataFrame(index=[str(t) for t in Attack.Mode], columns=RES_COLUMNS)
    df = do_metrics(df, full, 'full')

    meta = read_metadata(fold, delta)

    # check
    # calling the script with cached stuff makes examples from feature_selection
    # end up last in pruned (as they are re-inserted)
    # BUT we want them in front
    pruned_chunk1 = [i for i in pruned if i['index'] in meta['feats_ex']]
    pruned_chunk2 = [i for i in pruned if i['index'] not in meta['feats_ex']]
    pruned = pruned_chunk1 + pruned_chunk2

    if meta:

        # #pruned_indices = [i['index'] for i in pruned]
        # #full_only = [i for i in full if i['index'] not in pruned_indices]
        # #full_ = [i for i in full if i['index'] in pruned_indices]
        # full_only = [i for i in full if i['index'] in meta['feats_ex']]
        # full_ = [i for i in full if i['index'] not in meta['feats_ex']]

        # 2. pruned search: full_only + pruned
        # pruned_ = full_only.copy() + pruned.copy()
        # pruned_ = sorted(pruned_, key=lambda x: x['index'])  # (!) post-ICML (useless?)
        df = do_metrics(df, pruned, 'pruned')

        # extract dtrees time and sum it to the total
        # tab_meta = TABLES['metadata']
        # feats_time = tab_meta\
        #                 .select(tab_meta.feats_time)\
        #                 .where((tab_meta.delta==delta) & (tab_meta.fold==fold))\
        #                 .scalar()
        # meta = read_metadata(fold, delta)
        df.loc['pruned','time'] += meta['feats_time']

        # 3. mixed search
        # mixed = full_only.copy()
        calls_to_full, false_negs = 0, 0

        # print(len([i['index'] for i in full]))
        # print()
        # print(len([i['index'] for i in mixed]))
        # print()
        # print(len([i['index'] for i in pruned]))
        # print()
        # print(len(meta['feats_ex']))
        # exit()
       

        # with updated mixed procedure, this loop only counts fc, fn
        for f, p in zip(mixed, pruned):
            # print(f["index"], p["index"],  p['index'] in meta['feats_ex'])
            assert(f["index"] == p["index"])
            if p['index'] in meta['feats_ex']:  continue
            
            if p['result'] in [str(Attack.Result.UNSAT), str(Attack.Result.UNKNOWN)]:
                calls_to_full += 1
                #res = f.copy()
                #res["time"] = p["time"] + f["time"]

                if p["result"]==str(Attack.Result.UNSAT) and f["result"]==str(Attack.Result.SAT):
                    false_negs += 1
            #else:
                #res = p.copy()
            #mixed.append(res)

        df = do_metrics(df, mixed, 'mixed', calls_to_full, false_negs)
        df.loc['mixed','time'] += meta['feats_time']

        # test: study fold run times
        # test_study_fold_run_time(full, pruned_, meta)

        # test: see if the UNSATs are further away from the margin
        # test_unsats_distribution(full, meta, at)

    return df


def do_metrics(df, search, type_='full', calls_to_full=None, false_negs=None):

    stats = {
        "#sat": np.sum([1 if i['result']==str(Attack.Result.SAT) else 0 for i in search]),
        "#unsat": np.sum([1 if i['result']==str(Attack.Result.UNSAT) else 0 for i in search]),
        "#unk": np.sum([1 if i['result']==str(Attack.Result.UNKNOWN) else 0 for i in search]),
        "#miscl": np.sum([1 if i['result']==str(Attack.Result.MISCLASSIFIED) else 0 for i in search]),
        "#skipped": np.sum([1 if i['result']==str(Attack.Result.SKIPPED) else 0 for i in search]),
        "avg_sat_dist": np.mean([i['distance'] for i in search if i['distance'] is not None]),
        "avg_sat_output": np.mean([i['output'] for i in search if i['output'] is not None]),
        #"adv_acc": np.sum([1 if i['result']==str(Attack.Result.UNSAT) else 0 for i in search])\
        #            / len(search),
        "time": np.sum([i['time'] for i in search]),
        # "avg_time": np.mean([i['time'] for i in search]),
        # "stdev_time": np.std([i['time'] for i in search]),
    }
    
    if type_=='mixed' and calls_to_full is not None and false_negs is not None:
        stats['false_negs']=false_negs
        stats['full_calls']=calls_to_full
    elif type_=='mixed':    raise RuntimeError("Something wrong with mixed here.")

    df.loc[type_,:] = stats
    return df


# def average_metrics(folds_reports, columns):
#     print(f"\nAveraging results over {len(folds_reports)} folds.")

#     aggregated_results = {str(t): {k: [] for k in columns} for t in Attack.Mode}
    
#     for item in folds_reports:
#         for attack, row in item.iterrows():
#             for metric in item.columns:
#                 aggregated_results[attack][metric].append(row[metric])

#     df = pd.DataFrame(index=[str(t) for t in Attack.Mode], columns=columns)

#     for attack, metrics in aggregated_results.items():
#         for metric, values in metrics.items():
#             df.loc[attack, metric] = format_val_std(np.mean(values), np.round(np.std(values),3))
#     return df

def validate_metrics(df, num_examples):
    # check on adv_acc: #sat + #unsat + #unk + #miscl. + #skipped = tot. runs
    for i, row in df.iterrows():
        sum_ = row['#sat']+row['#unsat']+row['#unk']+row['#miscl']+row['#skipped']
        assert (sum_) == num_examples, \
            "AssertionError: #SAT + #UNSAT + #UNK + #MISCL + #SKIPPED != num_examples" \
            f" ({row['#sat']} + {row['#unsat']} + {row['#unk']} "\
            f"+ {row['#miscl']} + {row['#skipped']} = {sum_}, num_examples = {num_examples})"




