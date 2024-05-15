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
import verification

from sklearn.metrics import balanced_accuracy_score, root_mean_squared_error
     
@click.group()
def cli():
    pass

@cli.command("list")
@click.option("--cmd", type=click.Choice(["train", "compress"]), default="train")
@click.option("--linclf_type", type=click.Choice(["LogisticRegression", "Lasso"]),
              default="Lasso")
@click.option("--abserr", default=0.01)
@click.option("--seed", default=util.SEED)
def print_configs(cmd, linclf_type, abserr, seed):
    if cmd == "train":
        for dname in util.DNAMES:
            d = prada.get_dataset(dname, seed=seed, silent=True)

            for model_type in ["xgb"]:#, "dt"]:
                folds = [i for i in range(util.NFOLDS)]

                grid = d.paramgrid(fold=folds)

                for cli_param in grid:
                    print("python experiment.py train",
                          dname,
                          "--model_type", model_type,
                          "--linclf_type", linclf_type,
                          "--fold", cli_param["fold"],
                          "--seed", seed,
                          "--silent")
    elif cmd == "compress":
        for dname in util.DNAMES:
            d = prada.get_dataset(dname, seed=seed, silent=True)

            for model_type in ["xgb"]:#, "dt"]:
                folds = [i for i in range(util.NFOLDS)]

                grid = d.paramgrid(fold=folds)

                for cli_param in grid:
                    print("python experiment.py compress",
                          dname,
                          "--model_type", model_type,
                          "--linclf_type", linclf_type,
                          "--fold", cli_param["fold"],
                          "--abserr", abserr,
                          "--seed", seed,
                          "--silent")
    else:
        raise RuntimeError("dont know")


@cli.command("process_train")
@click.argument("fnames", nargs=-1)
def process_train_cmd(fnames):
    jsons = sum((util.read_json_printfile(fname) for fname in fnames), start=[])
    results = {}
    for j in jsons:
        dname = j["dname"]
        fold = j["fold"]
        key = util.get_key(j)
        models = j["models"]

        print(key, dname, fold)

        forkey = util.get_or_insert(results, key, lambda: {})
        fordname = util.get_or_insert(forkey, dname, lambda: {})
        onfront = util.pareto_front(models, mkey="mtest", skey="nnzleafs")

        for m, b in zip(models, onfront):
            params = m["params"]
            params_hash = util.params_hash(params)
            #print(params_hash, params)
            m["on_pareto_front"] = b
            forparams = util.get_or_insert(fordname, params_hash, lambda: {})
            if fold in m:
                print(f"overriding {fold} for {key} {dname} {params_hash}")
            forparams[fold] = m

    # We want to compress all param sets that in one of the folds was on the
    # pareto front, so we can average
    for key, forkey in results.items():
        for dname, fordname in forkey.items():
            for params_hash, folds in fordname.items():
                on_any_pareto_front = any(m["on_pareto_front"] for m in folds.values())
                for m in folds.values():
                    m["on_any_pareto_front"] = on_any_pareto_front

    util.write_train_results(results)


@cli.command("process_compress")
@click.argument("fnames", nargs=-1)
def process_compress_cmd(fnames):
    jsons = sum((util.read_json_printfile(fname) for fname in fnames), start=[])
    results = {}
    for j in jsons:
        dname = j["dname"]
        fold = j["fold"]
        key = util.get_key(j)

        print(key, dname, fold)

        forkey = util.get_or_insert(results, key, lambda: {})
        fordname = util.get_or_insert(forkey, dname, lambda: {})

        for m in j["models"]:
            params = m["params"]
            params_hash = util.params_hash(params)
            #print(params_hash, params)
            forparams = util.get_or_insert(fordname, params_hash, lambda: {})
            if fold in m:
                print(f"overriding {fold} for {key} {dname} {params_hash}")
            forparams[fold] = m

    util.write_compress_results(results)


@cli.command("plot_compress")
@click.argument("dname")
@click.option("-m", "--model_type", type=click.Choice(["xgb", "rf", "lgb", "dt"]),
              default="xgb")
@click.option("--linclf_type", type=click.Choice(["LogisticRegression", "Lasso"]),
              default="Lasso")
@click.option("--seed", default=util.SEED)
def plot_compress_cmd(dname, model_type, linclf_type, seed):
    if dname == "all":
        dnames = util.DNAMES
    else:
        dnames = [dname]

    key = util.get_key(model_type, linclf_type, seed)
    all_train_results = util.load_train_results()[key]
    all_compr_results = util.load_compress_results()[key]

    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(f"/tmp/figures/{key}.pdf") as pdf:
        plt.close('all')
        for dname in dnames:
            train_results = all_train_results[dname]
            compr_results = all_compr_results[dname]
            fig, ax, goodness_score = util.plot_pareto_fronts(
                dname, train_results, compr_results
            )

            fig.tight_layout()
            fig.savefig(f"/tmp/figures/{dname}-{key}.png")
            pdf.savefig(fig) 

            print(f"GOODNESS SCORE {dname} {goodness_score*100:.1f}%")

    if len(dnames) == 1:
        plt.show()
    


            




@cli.command("train")
@click.argument("dname")
@click.option("-m", "--model_type", type=click.Choice(["xgb", "rf", "lgb", "dt"]),
              default="xgb")
@click.option("--linclf_type", type=click.Choice(["LogisticRegression", "Lasso"]),
              default="Lasso")
@click.option("--fold", default=0)
@click.option("--seed", default=util.SEED)
@click.option("--silent", is_flag=True, default=False)
def train_cmd(dname, model_type, linclf_type, fold, seed, silent):
    d, dtrain, dvalid, dtest = util.get_dataset(dname, seed, linclf_type, fold, silent)
    model_class = d.get_model_class(model_type)

    hyperparam_time = time.time()
    param_dict = model_params.get_params(d, model_type)
    models = []

    for params in d.paramgrid(**param_dict):
        clf, train_time = dtrain.train(model_class, params)
        mtrain = dtrain.metric(clf)
        mtest  = dtest.metric(clf)
        mvalid = dvalid.metric(clf)

        at = veritas.get_addtree(clf, silent=True)
        nnzleafs = tree_compress.count_nnz_leafs(at)
        nleafs = at.num_leafs()

        res = util.HyperParamResult(
            at, train_time, params, mtrain, mvalid, mtest, nleafs, nnzleafs
        )
        models.append(res)

        del mtrain, mvalid, mtest  # don't use the wrong values accidentally
        del at, clf, train_time, params
    hyperparam_time = time.time() - hyperparam_time

    results = {
        # Experimental settings
        "cmd": "train",
        "date_time": util.nowstr(),
        "hostname": os.uname()[1],
        "dname": dname,
        "model_type": model_type,
        "fold": fold,
        "linclf_type": linclf_type,
        "seed": seed,
        "metric_name": d.metric_name,
    
        # Model stats, for xgb,dt,... model here
        "hyperparam_time": hyperparam_time,

        # Result for all the hyper parameter settings
        "models": [{
            "train_time": float(m.train_time),
            "params": m.params,
            "mtrain": float(m.mtrain),
            "mvalid": float(m.mvalid),
            "mtest": float(m.mtest),
            "ntrees": len(m.at),
            "nnodes": int(m.at.num_nodes()),
            "nleafs": int(m.nleafs),
            "nnzleafs": int(m.nnzleafs),
            "max_depth": int(m.at.max_depth()),
        } for m in models],
    
        # Data characteristics
        "ntrain": dtrain.X.shape[0],
        "nvalid": dvalid.X.shape[0],
        "ntest": dtest.X.shape[0],
    }
    if not silent:
        __import__('pprint').pprint(results)
    print(json.dumps(results))


@cli.command("compress")
@click.argument("dname")
@click.option("-m", "--model_type", type=click.Choice(["xgb", "rf", "lgb", "dt"]),
              default="xgb")
@click.option("--linclf_type", type=click.Choice(["LogisticRegression", "Lasso"]),
              default="Lasso")
@click.option("--fold", default=0)
@click.option("--abserr", default=0.01)
@click.option("--seed", default=util.SEED)
@click.option("--silent", is_flag=True, default=False)
@click.option("--plot", is_flag=True, default=False)
def compress_cmd(dname, model_type, linclf_type, fold, abserr, seed, silent, plot):
    d, dtrain, dvalid, dtest = util.get_dataset(dname, seed, linclf_type, fold, silent)
    model_class = d.get_model_class(model_type)
    max_rounds = 2

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

        #if not (tres["on_any_pareto_front"] and not tres["on_pareto_front"]):
        #    continue

        if not tres["on_any_pareto_front"]:
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

        data = tree_compress.Data(
                dtrain.X.to_numpy(), dtrain.y.to_numpy(),
                dtest.X.to_numpy(), dtest.y.to_numpy(),
                dvalid.X.to_numpy(), dvalid.y.to_numpy())

        compr = tree_compress.LassoCompress(
            data,
            at_orig,
            metric=mymetric,
            isworse=lambda v, ref: ref-v > abserr,
            linclf_type=linclf_type,
            seed=seed,
            silent=silent
        )
        compr.no_convergence_warning = True
        compr_time = time.time()
        at_compr = compr.compress(max_rounds=max_rounds)
        compr_time = time.time() - compr_time
        #compr.compress(max_rounds=1)
        #compr.linclf_type = "LogisticRegression"
        #at_compr = compr.compress(max_rounds=1)
        record = compr.records[-1]

        #if not silent:
        #    if at_compr.num_nodes() < 50:
        #        for i, t in zip(range(3), at_compr):
        #            print(t)
        #
        #if not silent:
        #    print(f"num_nodes {at.num_nodes()}->{at_compr.num_nodes()},",
        #          f"len {len(at)}->{len(at_compr)}",
        #          f"nnzleafs {compr.records[0].nnzleafs}->{record.nnzleafs}")
        #    print()

        compress_result = {
            "params": params,

            # Performance of the compressed model
            "compr_time": compr_time,
            "mtrain": float(record.mtrain),
            "mvalid": float(record.mvalid),
            "mtest": float(record.mtest),
            "ntrees": int(record.ntrees),
            "nnodes": int(record.nnodes),
            "nleafs": int(at_compr.num_leafs()),
            "nnzleafs": int(record.nnzleafs),
            "max_depth": int(at_compr.max_depth()),

            # Log the compression process
            "ntrees_rec": [int(r.ntrees) for r in compr.records],
            "nnodes_rec": [int(r.nnodes) for r in compr.records],
            "nnzleafs_rec": [int(r.nnzleafs) for r in compr.records],
            "mtrain_rec": [float(r.mtrain) for r in compr.records],
            "mvalid_rec": [float(r.mvalid) for r in compr.records],
            "mtest_rec": [float(r.mtest) for r in compr.records],
        }
        compress_results.append(compress_result)
        del compr

    results = {

        # Experimental settings
        "cmd": "compress",
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

    #if plot:
    #    import matplotlib.pyplot as plt

    #    mname = d.metric_name
    #    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
    #    ax0.semilogy(nnodes_rec, label="#nodes")
    #    ax0.semilogy(nnzleafs_rec, label="#non-zero leaves")
    #    ax0.legend()
    #    ax0.set_ylabel("num nodes/leaves")
    #    ax1.plot(mtrain_rec, label=f"{mname} train")
    #    ax1.plot(mtest_rec, label=f"{mname} test")
    #    ax1.plot(mvalid_rec, label=f"{mname} valid")
    #    ax1.axhline(y=mtest_rec[0], ls=":", color="gray")
    #    ax1.axhline(y=mtest_rec[0] - abserr,
    #                ls="-.", color="gray")
    #    ax1.legend()
    #    ax1.set_xlabel("iteration")
    #    ax1.set_ylabel(mname)
    #    fig.suptitle(dname)

    #    #config = veritas.Config(veritas.HeuristicType.MAX_OUTPUT)
    #    #config.stop_when_optimal = False
    #    #search = config.get_search(at_compr)
    #    #while search.steps(1000) != veritas.StopReason.NO_MORE_OPEN \
    #    #        and search.time_since_start() < 5.0:
    #    #    pass
    #    #    #print("Veritas", search.num_solutions(), search.num_open())
    #    #print("Veritas", search.num_solutions(), search.num_open())

    #    plt.show()


@cli.command("verification")
@click.argument("dname")
@click.option("-m", "--model_type", type=click.Choice(["xgb", "rf", "lgb", "dt"]),
              default="xgb")
@click.option("--linclf_type", type=click.Choice(["LogisticRegression", "Lasso"]),
              default="Lasso")
@click.option("--fold", default=0)
@click.option("--abserr", default=0.01)
@click.option("--timeout", default=15*60)
@click.option("--seed", default=util.SEED)
@click.option("--silent", is_flag=True, default=False)
def verification_cmd(
    dname, model_type, linclf_type, fold, abserr, timeout, seed, silent
):
    d, dtrain, dvalid, dtest = util.get_dataset(dname, seed, linclf_type, fold, silent)
    model_class = d.get_model_class(model_type)
    max_rounds = 2

    if linclf_type == "Lasso":  # binaryclf as a regr problem
        def mymetric(ytrue, ypred):
            return balanced_accuracy_score(ytrue > 0.0, ypred > 0.0)
    else:
        def mymetric(ytrue, ypred):
            return balanced_accuracy_score(ytrue, ypred)

    key = util.get_key(model_type, linclf_type, seed)
    train_results = util.load_train_results()[key][dname]
    compr_results = util.load_compress_results()[key][dname]

    #for params_hash, folds in train_results.items():

    params_hashes = [h for h, folds in train_results.items() if folds[fold]["on_pareto_front"]]
    params_hashes = sorted(params_hashes, key=lambda h: train_results[h][fold]["nnzleafs"])

    indices = np.unique(np.linspace(0, len(params_hashes)-1, 5).round().astype(int))
    params_hashes = [params_hashes[i] for i in indices]

    print(
        np.array(
            [
                [train_results[h][fold]["nnzleafs"] for h in params_hashes],
                [compr_results[h][fold]["nnzleafs"] for h in params_hashes],
            ]
        )
    )
    print(
        np.array(
            [
                [np.round(train_results[h][fold]["mtest"],3) for h in params_hashes],
                [np.round(compr_results[h][fold]["mtest"],3) for h in params_hashes],
            ]
        )
    )

    compress_results = []
    for params_hash in params_hashes:
        tres = train_results[params_hash][fold]
        cres = compr_results[params_hash][fold]
        params = tres["params"]

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

        data = tree_compress.Data(
                dtrain.X.to_numpy(), dtrain.y.to_numpy(),
                dtest.X.to_numpy(), dtest.y.to_numpy(),
                dvalid.X.to_numpy(), dvalid.y.to_numpy())

        compr = tree_compress.LassoCompress(
            data,
            at_orig,
            metric=mymetric,
            isworse=lambda v, ref: ref-v > abserr,
            linclf_type=linclf_type,
            seed=seed,
            silent=silent
        )
        compr.no_convergence_warning = True
        compr_time = time.time()
        at_compr = compr.compress(max_rounds=1) # !!!!!!!!!!!!!!
        compr_time = time.time() - compr_time
        record = compr.records[-1]


        ## VERIFICATION: (1) HOW MANY OCs?
        nocs_orig, nocs_orig_time, nocs_orig_timeout = verification.count_ocs(
            at_orig, timeout
        )
        nocs_compr, nocs_compr_time, nocs_compr_timeout = verification.count_ocs(
            at_compr, timeout
        )

        ## VERIFICATION: (2) Empricial robustness (exact + approx)
        n = 100
        rob_orig_exact, rob_orig_exact_time = verification.emp_robustness(
            at_orig, dtest.X, dtest.y, n, exact=True
        )
        rob_orig_approx, rob_orig_approx_time = verification.emp_robustness(
            at_orig, dtest.X, dtest.y, n, exact=False
        )
        rob_compr_exact, rob_compr_exact_time = verification.emp_robustness(
            at_compr, dtest.X, dtest.y, n, exact=True
        )
        rob_compr_approx, rob_compr_approx_time = verification.emp_robustness(
            at_compr, dtest.X, dtest.y, n, exact=False
        )

        compress_result = {
            "params": params,

            # Performance of the compressed model
            "compr_time": compr_time,
            "mtrain": float(record.mtrain),
            "mvalid": float(record.mvalid),
            "mtest": float(record.mtest),
            "ntrees": int(record.ntrees),
            "nnodes": int(record.nnodes),
            "nleafs": int(at_compr.num_leafs()),
            "nnzleafs": int(record.nnzleafs),
            "max_depth": int(at_compr.max_depth()),


            # Verification
            "verification": {
                "orig": {
                    "nocs": nocs_orig,
                    "nocs_time": nocs_orig_time,
                    "nocs_timeout": nocs_orig_timeout,
                    "exact_emp_rob": rob_orig_exact,
                    "exact_emp_rob_time":rob_orig_exact_time,
                    "approx_emp_rob": rob_orig_approx,
                    "approx_emp_rob_time": rob_orig_approx_time,
                },
                "compr": {
                    "nocs": nocs_compr,
                    "nocs_time": nocs_compr_time,
                    "nocs_timeout": nocs_compr_timeout,
                    "exact_emp_rob": rob_compr_exact,
                    "exact_emp_rob_time":rob_compr_exact_time,
                    "approx_emp_rob": rob_compr_approx,
                    "approx_emp_rob_time": rob_compr_approx_time,
                }
            }
        }
        compress_results.append(compress_result)
        del compr

        break

    results = {

        # Experimental settings
        "cmd": "compress",
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
