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

@cli.command("list")
@click.option("--cmd", type=click.Choice(["train", "compress"]), default="train")
@click.option("--linclf_type", type=click.Choice(["LogisticRegression", "Lasso"]),
              default="Lasso")
@click.option("--abserr", default=0.01)
@click.option("--seed", default=util.SEED)
def print_configs(cmd, linclf_type, abserr, seed):
    if cmd == "train":
        for dname in util.DNAMES_SUBSUB:
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
        for dname in util.DNAMES_SUBSUB:
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
@click.option("--nselected", default=5)
def process_train_cmd(fnames, nselected):
    jsons = sum((util.read_json_printfile(fname) for fname in fnames), start=[])
    results = {}
    for j in jsons:
        dname = j["dname"]
        fold = j["fold"]
        key = util.get_key(j)
        models = j["models"]

        if dname not in util.DNAMES_SUBSUB:
            continue

        forkey = util.get_or_insert(results, key, lambda: {})
        fordname = util.get_or_insert(forkey, dname, lambda: {})

        xs = np.array([m["nleafs"] for m in models])
        ys = np.array([m["mvalid"] for m in models])
        onfront, onhull = util.pareto_front_xy(xs, ys)

        print(key, dname, fold, "#onhull", onhull.sum(), onfront.sum())

        #if fold == 0:
        #    import matplotlib.pyplot as plt
        #    fig, ax = plt.subplots()
        #    ax.set_title(dname)
        #    ax.plot(xs, ys, ".")
        #    ax.plot(xs[onfront], ys[onfront], "o")
        #    ax.plot(xs[onhull], ys[onhull], "x")
        #    plt.show()

        for m, b, h in zip(models, onfront, onhull):
            params = m["params"]
            params_hash = util.params_hash(params)
            #print(params_hash, params)
            m["on_pareto_front"] = b
            m["on_chull_front"] = h
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
                on_any_chull_front = any(m["on_chull_front"] for m in folds.values())
                for m in folds.values():
                    m["on_any_pareto_front"] = on_any_pareto_front
                    m["on_any_chull_front"] = on_any_chull_front
                    m["selected"] = False

    # Figure out which parameter sets to use
    # (1) on the convex hulll front in at least one of the folds
    # (2) representative of the full model size range
    for key, forkey in results.items():
        for dname, fordname in forkey.items():
            hashes_onhull = {
                params_hash
                for params_hash, folds in fordname.items()
                if folds[0]["on_any_chull_front"]
            }

            def mean_mvalid(h): 
                return np.mean([m["mvalid"] for m in fordname[h].values()])
            def mean_nleafs(h): 
                return np.mean([m["nleafs"] for m in fordname[h].values()])

            seen = set()
            hs = []
            for h in sorted(hashes_onhull, key=mean_mvalid, reverse=True):
                value = np.mean([m["nleafs"]/2 for m in fordname[h].values()])
                value = int(np.round(value))
                if value not in seen:
                    hs.append(h)
                    seen.add(value)

            hs = sorted(hs, key=mean_nleafs)
            hs = [hs[i] for i in np.linspace(0, len(hs)-1, min(nselected, len(hs))).round().astype(int)]

            for h in hs:
                for fold in fordname[h].keys():
                    fordname[h][fold]["selected"] = True

            print(dname, "on_any_pare", sum(folds[1]["on_any_pareto_front"] for folds in fordname.values()))
            print(" " * len(dname), "on_any_hull", sum(folds[1]["on_any_chull_front"] for folds in fordname.values()))
            print(" " * len(dname), "   selected", len(hs))
            print("nleafs", np.round([np.mean([m["nleafs"] for m in
                                               fordname[h].values()]) for h in
                                      hs], 0))
            print("nest  ", np.round([np.mean([m["params"]["n_estimators"] for
                                               m in fordname[h].values()]) for
                                      h in hs], 0))
            print("depth ", np.round([np.mean([m["params"]["max_depth"] for m
                                               in fordname[h].values()]) for h
                                      in hs], 0))


            #import matplotlib.pyplot as plt
            #fig, ax = plt.subplots()
            #ax.set_title(dname)
            #xs = np.array([np.mean([m["nleafs"] for m in fordname[h].values()]) for h in fordname.keys()])
            #ys = np.array([np.mean([m["mtest"] for m in fordname[h].values()]) for h in fordname.keys()])
            #ax.plot(xs, ys, ".")
            #xs = np.array([np.mean([m["nleafs"] for m in fordname[h].values()]) for h in hs])
            #ys = np.array([np.mean([m["mtest"] for m in fordname[h].values()]) for h in hs])
            #ax.plot(xs, ys, "x")
            #plt.show()

    # Write out to file
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

        if "abserr" in j:
            name = f'{j["abserr"]*1000:03.0f}'
        elif j["cmd"].startswith("lr_"):
            name = j["cmd"]
        else:
            raise RuntimeError(f"don't know {j['cmd']=}")

        print(key, dname, fold, name)

        forkey = util.get_or_insert(results, key, lambda: {})
        fordname = util.get_or_insert(forkey, dname, lambda: {})

        for m in j["models"]:
            params = m["params"]
            params_hash = util.params_hash(params)
            #print(params_hash, params)
            forparams = util.get_or_insert(fordname, params_hash, lambda: {})
            forabserr = util.get_or_insert(forparams, name, lambda: {})
            if fold in forabserr:
                print(f"overriding {fold} for {key} {dname} {name} {params_hash[:6]}")
            forabserr[fold] = m

    #__import__('pprint').pprint(results)

    util.write_compress_results(results)


@cli.command("plot_compress")
@click.argument("dname")
@click.option("-m", "--model_type", type=click.Choice(["xgb", "rf", "lgb", "dt"]),
              default="xgb")
@click.option("--linclf_type", type=click.Choice(["LogisticRegression", "Lasso"]),
              default="Lasso")
@click.option("--seed", default=util.SEED)
def plot_compress_cmd(dname, model_type, linclf_type, seed):
    key = util.get_key(model_type, linclf_type, seed)
    all_train_results = util.load_train_results()[key]
    all_compr_results = util.load_compress_results()[key]

    if dname == "all":
        dnames = all_compr_results.keys()
    else:
        dnames = [dname]


    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    #with PdfPages(f"/tmp/figures/{key}-nnzleafs.pdf") as pdf:
    #    for dname in dnames:
    #        plt.close('all')
    #        train_results = all_train_results[dname]
    #        compr_results = all_compr_results[dname]
    #        fig, ax, goodness_score = util.plot_it_nogekeer(
    #            dname, train_results, compr_results, nnz=True
    #        )

    #        fig.tight_layout()
    #        fig.savefig(f"/tmp/figures/{dname}-{key}-nnz.png")
    #        pdf.savefig(fig) 

    #        print(f"GOODNESS SCORE {dname} {goodness_score*100:.1f}%")

    #with PdfPages(f"/tmp/figures/{key}-nleafs.pdf") as pdf:
    if True:
        for dname in dnames:
            plt.close('all')
            train_results = all_train_results[dname]
            compr_results = all_compr_results[dname]
            fig, ax, goodness_score = util.plot_it_nogekeer(
                dname, train_results, compr_results, nnz=False
            )

            #fig.savefig(f"/tmp/figures/{dname}-{key}.png")
            #pdf.savefig(fig) 

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

            ## Log the compression process
            #"ntrees_rec": [int(r.ntrees) for r in compr.records],
            #"nnodes_rec": [int(r.nnodes) for r in compr.records],
            #"nnzleafs_rec": [int(r.nnzleafs) for r in compr.records],
            #"mtrain_rec": [float(r.mtrain) for r in compr.records],
            #"mvalid_rec": [float(r.mvalid) for r in compr.records],
            #"mtest_rec": [float(r.mtest) for r in compr.records],
        }
        compress_results.append(compress_result)
        del compr

    results = {

        # Experimental settings
        "cmd": f"compress{abserr*1000:03.0f}",
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
