import os
import json
import time
import click
import veritas
import numpy as np

import prada
import model_params
import util
import tree_compress

from sklearn.metrics import balanced_accuracy_score, root_mean_squared_error

@click.group()
def cli():
    pass

@cli.command("compress")
@click.argument("dname")
@click.option("--fold", default=0)
@click.option("--abserr", default=0.01)
@click.option("--seed", default=util.SEED)
@click.option("--silent", is_flag=True, default=False)
def compress_cmd(dname, fold, abserr, seed, silent):
    # fixed parameters for workshop paper
    model_type = "xgb"
    linclf_type = "Lasso"
    max_rounds = 2

    d, dtrain, dvalid, dtest = util.get_dataset(dname, seed, linclf_type, fold, silent)
    model_class = d.get_model_class(model_type)

    if linclf_type == "Lasso":  # binaryclf as a regr problem
        def mymetric(ytrue, ypred):
            return balanced_accuracy_score(ytrue > 0.0, ypred > 0.0)
    else:
        def mymetric(ytrue, ypred):
            return balanced_accuracy_score(ytrue, ypred)

    key = util.get_key(model_type, linclf_type, seed)
    train_results = util.load_train_results()[key][dname]

    compress_results = []
    for params_hash, folds in train_results.items():
        tres = folds[fold]
        params = tres["params"]

        if not tres["selected"]:
            continue

        # Retrain the model
        clf, train_time = dtrain.train(model_class, params)
        mtrain = dtrain.metric(clf)
        mvalid = dvalid.metric(clf)
        mtest = dtest.metric(clf)
        at_orig = veritas.get_addtree(clf, silent=True)

        if not silent:
            print(f"{model_type} {d.metric_name}:")
            print(
                f"    RETRAINED MODEL: mtr {mtrain:.3f} mva {mvalid:.3f} mte {mtest:.3f}",
                f"in {train_time:.2f}s",
            )
            print(
                f"     PREVIOUS MODEL: mtr {tres['mtrain']:.3f} mva {tres['mvalid']:.3f}",
                f"mte {tres['mtest']:.3f}",
                f"in {tres['train_time']:.2f}s",
                "!! this should be the same !!"
            )

        assert np.abs(mtrain - tres["mtrain"]) < 1e-5
        assert np.abs(mvalid - tres["mvalid"]) < 1e-5
        assert np.abs(mtest - tres["mtest"]) < 1e-5

        ## RETRAINING XGB but now with reg_alpha value
        num_leaves_before = at_orig.num_leafs()

        at_best = at_orig
        smallest = num_leaves_before
        accuratest = mvalid
        best_alpha = 0.0

        alphas = np.power(10, np.linspace(-1, 4, 20))
        compr_time = time.time()
        for alpha in alphas:
            params_compr = params.copy()
            params_compr["reg_alpha"] = alpha

            clf, fit_time = dtrain.train(model_class, params_compr)
            mtrain_compr = dtrain.metric(clf)
            mvalid_compr = dvalid.metric(clf)
            at_compr = veritas.get_addtree(clf, silent=True)
            
            num_leaves_after = at_compr.num_leafs()

            s = "bad"
            if mvalid - mvalid_compr < abserr:
                s = "good enough"
                if (smallest > num_leaves_after):
                    s = "good enough smallest!"
                    at_best = at_compr
                    smallest = num_leaves_after
                    best_alpha = alpha
                elif (smallest == num_leaves_after) and (accuratest < mvalid_compr):
                    s = "good enough accuratest!"
                    at_best = at_compr
                    smallest = num_leaves_after
                    best_alpha = alpha

            if not silent:
                print(f"{alpha:7.1f} {mtrain_compr:.3f} {mvalid_compr:.3f} {num_leaves_after}",
                      f"{num_leaves_before-num_leaves_after} {s}")

        compr_time = time.time() - compr_time
        mtrain_compr = dtrain.metric(at_best)
        mvalid_compr = dvalid.metric(at_best)
        mtest_compr = dtest.metric(at_best)

        if not silent:
            print("mtrain", f"{mtrain:.3f} -> {mtrain_compr:.3f}")
            print("mvalid", f"{mvalid:.3f} -> {mvalid_compr:.3f}")
            print(" mtest", f"{mtest:.3f} -> {mtest_compr:.3f}")
            print("ntrees", f"{len(at_orig):5d} -> {len(at_best):5d}")
            print("nleafs", f"{at_orig.num_leafs():5d} -> {at_best.num_leafs():5d}")

        compress_result = {
            "params": params,
            "best_alpha": best_alpha,

            # Performance of the compressed model
            "compr_time": compr_time,
            "mtrain": float(mtrain_compr),
            "mvalid": float(mvalid_compr),
            "mtest": float(mtest_compr),
            "ntrees": int(len(at_best)),
            "nnodes": int(at_best.num_nodes()),
            "nleafs": int(at_best.num_leafs()),
            "nnzleafs": int(tree_compress.count_nnz_leafs(at_best)),
            "max_depth": int(at_best.max_depth()),

            ## Log the compression process
            #"ntrees_rec": [int(r.ntrees) for r in compr.records],
            #"nnodes_rec": [int(r.nnodes) for r in compr.records],
            #"nnzleafs_rec": [int(r.nnzleafs) for r in compr.records],
            #"mtrain_rec": [float(r.mtrain) for r in compr.records],
            #"mvalid_rec": [float(r.mvalid) for r in compr.records],
            #"mtest_rec": [float(r.mtest) for r in compr.records],
        }
        compress_results.append(compress_result)

    results = {

        # Experimental settings
        "cmd": f"xgbl1{abserr*1000:03.0f}",
        "date_time": util.nowstr(),
        "hostname": os.uname()[1],
        "dname": dname,
        "model_type": model_type,
        "fold": fold,
        "linclf_type": linclf_type,
        "seed": seed,
        "metric_name": d.metric_name,

        # Compression params
        "abserr": abserr,
        "max_rounds": max_rounds,

        # Results for compression on all models on the pareto front
        "models": compress_results,

        # Data characteristics
        "ntrain": dtrain.X.shape[0],
        "nvalid": dvalid.X.shape[0],
        "ntest": dtest.X.shape[0],
    }
    if not silent:
        __import__('pprint').pprint(results)
    print(json.dumps(results))


if __name__ == "__main__":
    cli()
