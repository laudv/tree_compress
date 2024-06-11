import os
import json
import click
import veritas
import numpy as np
import time
import warnings
import prada
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
import torch
import random
import torch.nn as nn

import util
import tree_compress


@click.group()
def cli():
    pass

@cli.command("list")
@click.option("--cmd", type=click.Choice(["train", "compress"]), default="train")
@click.option("--penalty", type=click.Choice(["l1", "l2", "lrl1"]),
              default="l2")
@click.option("--seed", default=util.SEED)
def print_configs(cmd, penalty, seed):
    for dname in util.DNAMES_SUBSUB:
        d = prada.get_dataset(dname, seed=seed, silent=True)

        folds = [i for i in range(util.NFOLDS)]

        grid = d.paramgrid(fold=folds)

        for cli_param in grid:
            print("python can_experiment.py leaf_refine",
                  dname,
                  "--penalty", penalty,
                  "--fold", cli_param["fold"],
                  "--seed", seed,
                  "--silent")


@cli.command("leaf_refine")
@click.argument("dname")
@click.option("--penalty", type=click.Choice(["l1", "l2", "lrl1", "ic", "gr"]),
              default="l2")
@click.option("--fold", default=0)
@click.option("--abserr", default=0.01)
@click.option("--seed", default=util.SEED)
@click.option("--silent", is_flag=True, default=False)
def leaf_refine_cmd(dname, penalty, fold, abserr, seed, silent):
    # fixed parameters for workshop paper
    model_type = "xgb"
    linclf_type = "Lasso"

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    d, dtrain, dvalid, dtest = util.get_dataset(dname, seed, linclf_type, fold, silent)
    model_class = d.get_model_class(model_type)

    # if `linclf_type == Lasso`, the classification task is converted into a
    # regression task with -1 for the negative class and +1 for the positive
    # class. If `linclf_type == LogisticRegression`, it is normal binary
    # classification.
    key = util.get_key(model_type, linclf_type, seed)
    train_results = util.load_train_results()[key][dname]

    refine_results = []
    for params_hash, folds in train_results.items():
        tres = folds[fold]
        params = tres["params"]

        if not tres["selected"]:
            continue

        # Retrain the model
        clf, train_time = dtrain.train(model_class, params)
        mtrain = dtrain.metric(clf)
        mvalid = dvalid.metric(clf)
        mtest =  dtest.metric(clf)
        at_orig = veritas.get_addtree(clf, silent=True)
        accuratest = mvalid
        best_alpha = 0.0
        num_leaves_before = at_orig.num_leafs()
        smallest = num_leaves_before

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

        refine_time = time.time()
        if penalty == "lrl1":
            refiner = LRPlusL1Refiner()
        elif penalty == "l2" or penalty == "gr":
            refiner = LogisticRegression(
                penalty="l2",
                fit_intercept=True,
                solver="liblinear",
                dual=True,
                max_iter=10000,
                warm_start=False,
                random_state=1351,
            )
        else:
            refiner = LogisticRegression(
                penalty="l1",
                fit_intercept=True,
                solver="liblinear",
                max_iter=3000,
                warm_start=False,
                random_state=3582134,
            )

        sparse_train_x = transform_data_sparse(at_orig, dtrain.X)
        sparse_valid_x = transform_data_sparse(at_orig, dvalid.X)

        if penalty == "lrl1":

            #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            alpha_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.925,0.955,0.975,1]
            for alpha in alpha_list:
                refiner.set_params(at_orig, alpha)
                refiner.refine(50, sparse_train_x, dtrain.y)
                preds = refiner(torch.from_numpy(sparse_valid_x.todense())).detach().numpy()
                score = dvalid.metric(preds > 0.0)
                at_compr = at_orig.copy()
                at_compr = set_new_addtree(
                refiner.tree_weights.detach().numpy(),
                [w.detach().numpy() for w in refiner.leaf_weights],
                refiner.base_score.detach().numpy(),
                at_compr,
            )
                num_leaves_after = at_compr.num_leafs()
                if mvalid - score < abserr:
                    if (smallest > num_leaves_after):
                        smallest = num_leaves_after
                        best_alpha = alpha
                    elif (smallest == num_leaves_after) and (accuratest < score):
                        smallest = num_leaves_after
                        best_alpha = alpha

            refiner.set_params(at_orig, best_alpha)
            refiner.refine(50, sparse_train_x, dtrain.y)
            
            at_refined = at_orig.copy()
            at_refined = set_new_addtree(
                refiner.tree_weights.detach().numpy(),
                [w.detach().numpy() for w in refiner.leaf_weights],
                refiner.base_score.detach().numpy(),
                at_refined,
            )

        elif penalty == "gr":
            alpha_list = [10000, 1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
            for alpha in  alpha_list:
                refiner.set_params(C = 1/alpha)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    refiner.fit(sparse_train_x, dtrain.y)
                at_temp_refined = at_orig.copy()
                at_temp_refined = set_new_leaf_vals(at_temp_refined, refiner.intercept_[0], refiner.coef_[0])
                at_temp_pruned = veritas.AddTree(at_orig.num_leaf_values(), at_orig.get_type())
                for tree_index, t in enumerate(at_temp_refined):
                    nt = at_temp_pruned.add_tree()
                    gr_prune(t,t.root(),nt,nt.root(),0.1)
                score = dvalid.metric(at_temp_pruned)
                num_leaves_after = at_temp_pruned.num_leafs()
                if mvalid - score < abserr:
                    if (smallest > num_leaves_after):
                        smallest = num_leaves_after
                        best_alpha = alpha
                    elif (smallest == num_leaves_after) and (accuratest < score):
                        smallest = num_leaves_after
                        best_alpha = alpha

            if best_alpha != 0.0:
                refiner.set_params(C=1 / best_alpha)
                refiner.fit(sparse_train_x, dtrain.y)
                at_refined_temp = at_orig.copy()
                at_refined_temp = set_new_leaf_vals(
                    at_refined_temp, refiner.intercept_[0], refiner.coef_[0]
                )
                at_refined = veritas.AddTree(at_orig.num_leaf_values(), at_orig.get_type())
                for tree_index, t in enumerate(at_refined_temp):
                    nt = at_refined.add_tree()
                    gr_prune(t, t.root(), nt, nt.root(), 0.1)
            else:
                at_refined = at_orig.copy()

        elif penalty == "ic":
            # alpha here is number of expected trees in the ensemble
            ensamble_proba = np.asarray([(t.eval(dtrain.X).clip(-1,1) + 1) / 2 for t in at_orig])
            ensamble_proba = np.concatenate((1-ensamble_proba,ensamble_proba),axis=2)
            ic_list = individual_contribution(ensamble_proba, (np.asarray(dtrain.y) + 1) / 2)
            sorted_trees = [at_orig[i] for i in np.argsort(ic_list)]
            temp_at = veritas.AddTree(1, veritas.AddTreeType.REGR)
            temp_at.set_base_score(0,at_orig.get_base_score(0))
            for alpha in range(len(at_orig)):
                temp_at.add_tree(sorted_trees[alpha])
                score = dvalid.metric(temp_at)
                num_leaves_after = temp_at.num_leafs()
                if mvalid - score < abserr:
                    if (smallest > num_leaves_after):
                        smallest = num_leaves_after
                        best_alpha = alpha
                    elif (smallest == num_leaves_after) and (accuratest < score):
                        smallest = num_leaves_after
                        best_alpha = alpha

            at_refined = veritas.AddTree(1, veritas.AddTreeType.REGR)
            at_refined.set_base_score(0, at_orig.get_base_score(0))
            for alpha in range(int(best_alpha)):
                at_refined.add_tree(sorted_trees[alpha])

        elif penalty == "l1":
            alpha_list = [10000, 1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
            for alpha in  alpha_list:
                refiner.set_params(C = 1/alpha)
                #with warnings.catch_warnings():
                #    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                refiner.fit(sparse_train_x, dtrain.y)
                preds = refiner.predict(sparse_valid_x)
                score = dvalid.metric(preds > 0.0)
                at_temp = at_orig.copy()
                at_temp = set_new_leaf_vals(at_temp, refiner.intercept_[0], refiner.coef_[0])
                num_leaves_after = at_temp.num_leafs()
                if mvalid - score < abserr:
                    if (smallest > num_leaves_after):
                        smallest = num_leaves_after
                        best_alpha = alpha
                    elif (smallest == num_leaves_after) and (accuratest < score):
                        smallest = num_leaves_after
                        best_alpha = alpha

            if best_alpha != 0.0:
                refiner.set_params(C = 1/best_alpha)
                refiner.fit(sparse_train_x,dtrain.y)

                at_refined = at_orig.copy()
                at_refined = set_new_leaf_vals(at_refined, refiner.intercept_[0], refiner.coef_[0])
            else:
                at_refined = at_orig.copy()
        else:
            raise RuntimeError(f"I don't know {penalty}")
        
        refine_time = time.time() - refine_time
        mtrain_refined = dtrain.metric(at_refined)
        mvalid_refined = dvalid.metric(at_refined)
        mtest_refined = dtest.metric(at_refined)

        if not silent:
            print(" alpha", best_alpha)
            print("mtrain", f"{mtrain:.3f} -> {mtrain_refined:.3f}")
            print("mvalid", f"{mvalid:.3f} -> {mvalid_refined:.3f}")
            print(" mtest", f"{mtest:.3f} -> {mtest_refined:.3f}")
            print("ntrees", f"{len(at_orig):5d} -> {len(at_refined):5d}")
            print("nleafs", f"{at_orig.num_leafs():5d} -> {at_refined.num_leafs():5d}")
            print("   nnz", f"{tree_compress.count_nnz_leafs(at_orig):5d} ->",
                            f"{tree_compress.count_nnz_leafs(at_refined):5d}")

        refine_result = {
            "params": params,
            "best_alpha": best_alpha,

            # Performance of the compressed model
            "compr_time": refine_time,
            "mtrain": mtrain_refined,
            "mvalid": mvalid_refined,
            "mtest":  mtest_refined,
            "ntrees": len(at_refined),
            "nnodes": int(at_refined.num_nodes()),
            "nleafs": int(at_refined.num_leafs()),
            "nnzleafs": int(tree_compress.count_nnz_leafs(at_refined)),
            "max_depth": int(at_refined.max_depth()),
        }

        refine_results.append(refine_result)

    results = {
        # Experimental settings
        "cmd": f"can_{penalty}_{abserr*1000:03.0f}",
        "date_time": util.nowstr(),
        "hostname": os.uname()[1],
        "dname": dname,
        "model_type": model_type,
        "fold": fold,
        "linclf_type": linclf_type,
        "seed": seed,
        "metric_name": d.metric_name,

        # Any additional parameters
        #<param>: <value>
        "penalty": penalty,

        # Result for all the hyper parameter settings
        "models": refine_results,
    
        # Data characteristics
        "ntrain": dtrain.X.shape[0],
        "nvalid": dvalid.X.shape[0],
        "ntest": dtest.X.shape[0],
    }
    if not silent:
        __import__('pprint').pprint(results)
    print(json.dumps(results))


def gr_prune(t, n, tc, nc, threshold):
    if t.is_leaf(n):
        tc.set_leaf_value(nc, 0, t.get_leaf_value(n, 0))
    elif t.is_leaf(t.left(n)) and t.is_leaf(t.right(n)):
        # this is the interesting case
        llv = t.get_leaf_value(t.left(n), 0)
        rlv = t.get_leaf_value(t.right(n), 0)
        if np.sqrt((llv - rlv)**2) < threshold:
            tc.set_leaf_value(nc, 0, (llv+rlv)/2)
        else:
            split = t.get_split(n)
            tc.split(nc, split.feat_id, split.split_value)
            gr_prune(t, t.left(n), tc, tc.left(nc), threshold)
            gr_prune(t, t.right(n), tc, tc.right(nc), threshold)
    else:
        split = t.get_split(n)
        tc.split(nc, split.feat_id, split.split_value)
        gr_prune(t, t.left(n), tc, tc.left(nc), threshold)
        gr_prune(t, t.right(n), tc, tc.right(nc), threshold)

def transform_data_sparse(at, x):
    row_ind, col_ind = [], []
    num_rows = x.shape[0]
    num_cols = 0

    for tree_index, t in enumerate(at):
        leaf_ids = t.get_leaf_ids()
        num_leaves = len(leaf_ids)
        offset = num_cols
        num_cols += num_leaves

        leaf2index = np.zeros(t.num_nodes(), dtype=int)
        for cnt, id in enumerate(leaf_ids):
            leaf2index[id] = cnt

        row_ind += list(range(num_rows))
        col_ind += list(map(lambda u: offset + leaf2index[u], t.eval_node(x)))

    return csr_matrix((np.ones(len(row_ind), dtype=np.float64),
                              (row_ind, col_ind)), shape=(num_rows, num_cols))

def set_new_leaf_vals(at, base_score, new_leaf_vals):
    at.set_base_score(0, base_score)
    offset = 0
    for tree_index, t in enumerate(at):
        for count , leaf_id in enumerate(t.get_leaf_ids()):
            t.set_leaf_value(leaf_id, 0, new_leaf_vals[offset + count])
        offset += count + 1
    return at

def set_new_addtree(tree_weights, leaf_weights, base_score, at):
    new_at = veritas.AddTree(1, veritas.AddTreeType.REGR)
    new_at.set_base_score(0,base_score)
    for tree_index, t in enumerate(at):
        if tree_weights[tree_index] != 0:
            for count , leaf_id in enumerate(t.get_leaf_ids()):
                t.set_leaf_value(leaf_id, 0, leaf_weights[tree_index][count] * tree_weights[tree_index])
            new_at.add_tree(t)
    return new_at



def individual_contribution(ensemble_proba, target):
    """
    Compute the individual contributions of each classifier wrt. the entire ensemble. Return the negative contribution due to the minimization.

    Reference:
        Lu, Z., Wu, X., Zhu, X., & Bongard, J. (2010). Ensemble pruning via individual contribution ordering. Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 871–880. https://doi.org/10.1145/1835804.1835914
    """

    ic_list = []
    for i in range(len(ensemble_proba)):
        iproba = ensemble_proba[i,:,:]
        n = iproba.shape[0]

        predictions = iproba.argmax(axis=1)
        V = np.zeros(ensemble_proba.shape)
        idx = ensemble_proba.argmax(axis=2)
        V[np.arange(ensemble_proba.shape[0])[:,None],np.arange(ensemble_proba.shape[1]),idx] = 1
        V = V.sum(axis=0)

        IC = 0
        #V = all_proba.argmax(axis=2)
        #predictions = iproba.argmax(axis=1)
        #V = all_proba.sum(axis=0)#.argmax(axis=1)

        for j in range(n):
            if (predictions[j] == target[j]):                
                # case 1 (minority group)
                # label with majority votes on datapoint  = np.argmax(V[j, :]) 
                if(predictions[j] != np.argmax(V[j,:])):
                    IC = IC + (2*(np.max(V[j,:])) - V[j, predictions[j]])
                    
                else: # case 2 (majority group)
                    # calculate second largest nr of votes on datapoint i
                    sortedArray = np.sort(np.copy(V[j,:]))
                    IC = IC + (sortedArray[-2])
                    
            else:
                # case 3 (wrong prediction)
                IC = IC + (V[j,  int(target[j])]  -  V[j, predictions[j]] - np.max(V[j,:]) )
        ic_list.append(- 1.0 * IC)
    return ic_list



class LRPlusL1Refiner(nn.Module):
    def __init__(self):
        super(LRPlusL1Refiner, self).__init__()
        self.at = None
        self.leaf_weights = None
        self.tree_weights = None
        self.base_score = None
        self.alpha = None
        #self.sm = nn.Softmax(dim=1)

    def set_params(self,at,alpha):
        self.at = at
        torch_leafs = []
        torch_trees = []
        for tree_index, t in enumerate(at):
            leaf_vals = []
            for l_id in t.get_leaf_ids():
                leaf_vals.append(t.get_leaf_value(l_id,0))
            leaf_vals = np.array(leaf_vals)
            torch_leafs.append(nn.Parameter(torch.from_numpy(leaf_vals)))
            torch_trees.append(1)
        self.leaf_weights = nn.ParameterList(torch_leafs)
        self.tree_weights= nn.Parameter(torch.Tensor(torch_trees))
        self.base_score = nn.Parameter(torch.tensor(at.get_base_score(0)))
        self.alpha = alpha

    def forward(self, x):
        offset = 0
        y = 0
        for i, p in enumerate(self.leaf_weights):
            #y += torch.sparse.mm(x[:,offset:len(p)], p) * self.tree_weights[i]
            y += torch.matmul(x[:,offset:len(p)+offset],p) * self.tree_weights[i]
            offset += len(p)
        y += self.base_score
        return y
    
    def refine(self, epochs, X, Y, batch_size = 128):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        for epoch in range(epochs):
            mini_batches = create_mini_batches(X,Y, batch_size , True) 
            for x,y in mini_batches:
                optimizer.zero_grad()
                loss = criterion(self(x),y)
                loss.backward()
                optimizer.step()
                
                with torch.no_grad():
                    step_size = optimizer.param_groups[0]['lr']
                    proxed = psgd(self.tree_weights.numpy(),self.alpha,step_size)
                    self.tree_weights.copy_(torch.from_numpy(proxed))

def psgd(w, alpha, step_size):
    sign = np.sign(w)
    tmp_w = np.abs(w) - alpha*step_size
    tmp_w = sign*np.maximum(tmp_w,0)
    return tmp_w


def create_mini_batches(inputs, targets, batch_size, shuffle=False):
    """ Create an mini-batch like iterator for the given inputs / target / data. Shamelessly copied from https://stackoverflow.com/questions/38157972/how-to-implement-mini-batch-gradient-descent-in-python
    
    Parameters
    ----------
    inputs : array-like vector or matrix 
        The inputs to be iterated in mini batches
    targets : array-like vector or matrix 
        The targets to be iterated in mini batches
    batch_size : int
        The mini batch size
    shuffle : bool, default False
        If True shuffle the batches 
    """
    assert inputs.shape[0] == targets.shape[0]
    indices = np.arange(inputs.shape[0])
    if shuffle:
        np.random.shuffle(indices)
    
    start_idx = 0
    while start_idx < len(indices):
        if start_idx + batch_size > len(indices) - 1:
            excerpt = indices[start_idx:]
        else:
            excerpt = indices[start_idx:start_idx + batch_size]
        
        start_idx += batch_size

        #yield torch.sparse_coo_tensor(inputs[excerpt].nonzero(), inputs[excerpt].data, inputs[excerpt].shape), targets.iloc[excerpt]
        yield torch.from_numpy(inputs[excerpt].todense()), torch.tensor(targets.iloc[excerpt].values,dtype=torch.float64)


    


if __name__ == "__main__":
    cli()
