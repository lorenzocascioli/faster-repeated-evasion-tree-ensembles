import click
import prada
from attacks import *
from datasets import *
from db import *
from metrics import *
from utils import *

import warnings; warnings.filterwarnings("ignore")

@click.command()
@click.argument("dname")
@click.option("--seed", default=SEED)
@click.option("--nfolds", default=NFOLDS) 
@click.option("--fold", default=0) 
@click.option("-m", "--model_type", type=click.Choice(["xgb", "rf", "groot"]), default="xgb")
@click.option("-a", "--attack_type", type=click.Choice(["ver", "kan"]), default="ver")
@click.option("-N", "--num_examples", default=10000) 
@click.option("--feats_n", default=500)
@click.option("--fnr", default=FNR)
@click.option("--delta", default=None)
@click.option("--force_new", default=False) 
@click.option("--pruned_only", default=False) 
@click.option("--debug", default=None)
@click.option("--cache", default=False)
def main(dname, seed, nfolds, fold, model_type, attack_type, num_examples,\
			feats_n, fnr, delta, force_new, pruned_only, debug, cache):


	########## 1. LOAD DATASET + LOAD/TRAIN MODEL + SETUP DB ##########

	# d, params = dataset
	d = prada.get_dataset(dname, seed=seed)

	d.load_dataset()
	d.minmax_normalize()
	d.astype(np.float32); d.astype(veritas.FloatT) # for libsvm datasets

	params = THE_PARAMS[d.name()]
	delta = params['delta'][model_type] if delta is None else float(delta)

	if attack_type=='ver':	attack_type=Attack.Type.Veritas
	elif attack_type=='kan':	attack_type=Attack.Type.Kantchelian
	else: raise RuntimeError("Unsupported attack type!")
	
	print(f"\n***** Dataset: {d.name()} - model: {model_type} - attack: {attack_type}",\
			f" - delta: {delta} - fold: {fold} *****\n")

	# load model
	model, meta, at = get_model(d, fold, model_type, params, attack_type, cache)
	
	# test: count num. leaves
	# print(f'Average no. of leaves in full model: {count_avg_model_size(at)}')

	# database setup (peewee)
	connect_and_setup_db(d, fold, model_type, attack_type, seed, \
							num_examples, feats_n, fnr, cache)
	if force_new:	delete_fold_data(fold, pruned_only)
	if cache:	check_hostname()

	########## 2. CHECK WHAT'S CACHED, EXTRACT INDICES ##########

	# possibly read which experiments have already been run
	cached = check_cached_runs(fold, delta, cache)

	# if debug mode, just deal with the one example of interest
	if debug is not None:
		indices = debug_index(d, at, debug)
	else:
		# get test set indices
		# indices = get_N_test_example_indices(d, fold, num_examples)
		# indices = exclude_misclassified_indices(d, at, indices, fold, \
		# 											num_examples, delta, cached)
        # no more misclassified examples, we immediately drop those
		indices = get_N_correct_test_example_indices(d, at, fold, num_examples)

	# guard: order of magnit. of min dist. btw two split_values of the same feat.
    # (so that when we do split_value-guard we do not cross multiple splits!)
	guard = extract_guard(at)

	########## 3. PERFORM SEARCHES ##########

	for mode in Attack.Mode:
		if mode==Attack.Mode.Full and pruned_only:  continue

		t = time.time()

		get_adversarial_examples(d, at, indices, fold, delta, attack_type,\
									model_type, mode, num_examples, feats_n, \
									fnr, guard, cache, cached, debug)

		print(f"overall time to do {mode}: {time.time()-t:.2f} s\n")
		
	if debug is not None:   exit()

	# extract fold summary
	t = time.time()
	fold_df = gather_results(delta, fold, at=at)
	#validate_metrics(fold_df, num_examples)
	print(f"\nResults - fold {fold}\n", fold_df)
	print(f"time to gather results: {time.time()-t:.2f} s")

    # average folds
    #avg_df = calculate_metrics(num_examples, delta, experiment_type)
    #print(avg_df)

	close_db_connection()


if __name__ == '__main__':
    main()