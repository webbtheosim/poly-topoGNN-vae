import os
import glob
import pickle

import numpy as np
import sklearn.metrics as skm

from tqdm import tqdm
from model_utils import get_spec, train_vae, latent_model
from data_utils import load_data

WEIGHT_DIR = '/scratch/gpfs/sj0161/topo_result/' # change to your directory
DATA_DIR = '/scratch/gpfs/sj0161/topo_data/' # change to your directory

def get_metrics(model, enc_type, if_reg, if_cls,
                x_train, y_train, c_train, l_train,
                x_valid, y_valid, c_valid, l_valid,
                x_test, y_test, c_test, l_test,
                n_repeat=5, if_bacc=False):
    
    rmse = []
    r2 = []
    f1 = []
    acc = []
    bacc = []
    x_pred_trains = []
    x_pred_valids = []
    x_pred_tests = []
    y_pred_trains = []
    y_pred_valids = []
    y_pred_tests = []
    c_pred_trains = []
    c_pred_valids = []
    c_pred_tests = []
    
    for i in range(n_repeat):
        if enc_type == 'cnn' or enc_type == 'dnn':
            in_train = x_train
            in_valid = x_valid
            in_test = x_test
        elif enc_type == 'gnn':
            in_train = [x_train, x_train]
            in_valid = [x_valid, x_valid]
            in_test = [x_test, x_test]
        elif enc_type == 'desc_dnn':
            in_train = l_train
            in_valid = l_valid
            in_test = l_test
        elif enc_type == 'desc_gnn':
            in_train = [[x_train, x_train], l_train]
            in_valid = [[x_valid, x_valid], l_valid]
            in_test = [[x_test, x_test], l_test]
        
        if not if_reg and not if_cls:
            x_pred_test = model.predict(in_test, verbose=0)
            x_pred_train = model.predict(in_train, verbose=0)
            x_pred_valid = model.predict(in_valid, verbose=0)
            x_pred_trains.append(x_pred_train)
            x_pred_valids.append(x_pred_valid)
            x_pred_tests.append(x_pred_test)
        elif if_reg and not if_cls:
            x_pred_test, y_pred_test = model.predict(in_test, verbose=0)
            x_pred_train, y_pred_train = model.predict(in_train, verbose=0)
            x_pred_valid, y_pred_valid = model.predict(in_valid, verbose=0)
            x_pred_trains.append(x_pred_train)
            x_pred_valids.append(x_pred_valid)
            x_pred_tests.append(x_pred_test)
            y_pred_tests.append(y_pred_test)
            y_pred_trains.append(y_pred_train)
            y_pred_valids.append(y_pred_valid)
        elif not if_reg and if_cls:
            x_pred_test, c_pred_test = model.predict(in_test, verbose=0)
            x_pred_train, c_pred_train = model.predict(in_train, verbose=0)
            x_pred_valid, c_pred_valid = model.predict(in_valid, verbose=0)
            x_pred_trains.append(x_pred_train)
            x_pred_valids.append(x_pred_valid)
            x_pred_tests.append(x_pred_test)
            c_pred_tests.append(c_pred_test)
            c_pred_trains.append(c_pred_train)
            c_pred_valids.append(c_pred_valid)
        else:
            x_pred_test, y_pred_test, c_pred_test = model.predict(in_test, verbose=0)
            x_pred_train, y_pred_train, c_pred_train = model.predict(in_train, verbose=0)
            x_pred_valid, y_pred_valid, c_pred_valid = model.predict(in_valid, verbose=0)
            x_pred_trains.append(x_pred_train)
            x_pred_valids.append(x_pred_valid)
            x_pred_tests.append(x_pred_test)
            c_pred_tests.append(c_pred_test)
            c_pred_trains.append(c_pred_train)
            c_pred_valids.append(c_pred_valid)
            y_pred_tests.append(y_pred_test)
            y_pred_trains.append(y_pred_train)
            y_pred_valids.append(y_pred_valid)
        
        if if_cls:
            c_pred_test = np.argmax(c_pred_test, axis=1)
            c_pred_train = np.argmax(c_pred_train, axis=1)
            c_pred_valid = np.argmax(c_pred_valid, axis=1)
        
        rmse.append([])
        rmse[i].append(skm.mean_squared_error(y_train, y_pred_train) ** 0.5)
        rmse[i].append(skm.mean_squared_error(y_valid, y_pred_valid) ** 0.5)
        rmse[i].append(skm.mean_squared_error(y_test, y_pred_test) ** 0.5)
        
        r2.append([])
        r2[i].append(skm.r2_score(y_train, y_pred_train))
        r2[i].append(skm.r2_score(y_valid, y_pred_valid))
        r2[i].append(skm.r2_score(y_test, y_pred_test))
        
        f1.append([])
        acc.append([])
        
        if if_cls:
            f1[i].append(skm.f1_score(c_train, c_pred_train, average='weighted'))
            f1[i].append(skm.f1_score(c_valid, c_pred_valid, average='weighted'))
            f1[i].append(skm.f1_score(c_test, c_pred_test, average='weighted'))
            acc[i].append(skm.accuracy_score(c_train, c_pred_train))
            acc[i].append(skm.accuracy_score(c_valid, c_pred_valid))
            acc[i].append(skm.accuracy_score(c_test, c_pred_test))
        
        if if_bacc:
            xt1 = x_train.ravel()
            xp1 = np.round(x_pred_train.ravel())
            xt2 = x_valid.ravel()
            xp2 = np.round(x_pred_valid.ravel())
            xt3 = x_test.ravel()
            xp3 = np.round(x_pred_test.ravel())
            bacc.append([])
            bacc[i].append(skm.balanced_accuracy_score(xt1, xp1))
            bacc[i].append(skm.balanced_accuracy_score(xt2, xp2))
            bacc[i].append(skm.balanced_accuracy_score(xt3, xp3))
    
    rmse = np.array(rmse)
    r2 = np.array(r2)
    f1 = np.array(f1)
    bacc = np.array(bacc)
    acc = np.array(acc)
    rmse_m = None
    r2_m = None
    f1_m = None
    acc_m = None
            
    to_str = lambda x: f"{np.mean(x):0.2f}+/-{np.std(x):0.2f}"
    print(f"RMSE: Train {to_str(rmse[:,0])} Valid {to_str(rmse[:,1])} Test {to_str(rmse[:,2])}")
    rmse_m = rmse.mean(axis=0)
    if if_reg:
        print(f"R2:   Train {to_str(r2[:,0])} Valid {to_str(r2[:,1])} Test {to_str(r2[:,2])}")
        r2_m = r2.mean(axis=0)
    if if_cls:
        print(f"F1:   Train {to_str(f1[:,0])} Valid {to_str(f1[:,1])} Test {to_str(f1[:,2])}")
        f1_m = f1.mean(axis=0)
    if if_bacc:
        print(f"BACC: Train {to_str(bacc[:,0])} Valid {to_str(bacc[:,1])} Test {to_str(bacc[:,2])}")
    
    if if_reg and if_cls:
        train_out = (x_pred_trains, y_pred_trains, c_pred_trains)
        valid_out = (x_pred_valids, y_pred_valids, c_pred_valids)
        test_out = (x_pred_tests, y_pred_tests, c_pred_tests)
    elif if_reg and not if_cls:
        train_out = (x_pred_trains, y_pred_trains)
        valid_out = (x_pred_valids, y_pred_valids)
        test_out = (x_pred_tests, y_pred_tests)
    elif not if_reg and if_cls:
        train_out = (x_pred_trains, c_pred_trains)
        valid_out = (x_pred_valids, c_pred_valids)
        test_out = (x_pred_tests, c_pred_tests)
    else:
        train_out = x_pred_trains
        valid_out = x_pred_valids
        test_out = x_pred_tests
        
    
    return train_out, valid_out, test_out, bacc, rmse, r2, f1, acc


def get_val_metrics(encoder):
    """
    Get validation metrics for a specific encoder.

    Args:
        encoder (str): The encoder identifier.

    Returns:
        tuple: A tuple containing arrays of various metrics, including elbos, baccs, kls, cls, rls, rec, rmses, r2s, f1s, and selected files.
    """
    ((x_train, y_train, c_train, l_train, graph_train),
    (x_valid, y_valid, c_valid, l_valid, graph_valid),
    (x_test, y_test, c_test, l_test, graph_test),
    NAMES, SCALER, LE) = load_data(os.path.join(DATA_DIR, 'rg2.pickle'), fold=0, if_validation=True)

    graph_all = np.concatenate((graph_train, graph_valid, graph_test))

    files1 = sorted(glob.glob(WEIGHT_DIR + f"/{encoder}*True*True*.pickle"))
    files2 = sorted(glob.glob(WEIGHT_DIR + f"/{encoder}*True*True*metric.pickle"))
    files = list(set(files1) - set(files2))

    elbos = []
    kls = []
    baccs = []
    cls = []
    rls = []
    files_select = []
    rmses = []
    r2s = []
    f1s = []

    for file in tqdm(files, total=len(files)):
        with open(file, 'rb') as handle:
            hist = pickle.load(handle)
        if "val_decoder_acc" in file:
            idx = np.argmax(hist["val_decoder_acc"])
        elif "val_decoder_loss" in file:
            idx = np.argmin(hist["val_decoder_loss"])
        elif "val_loss" in file:
            idx = np.argmin(hist["val_loss"])
        
        if os.path.exists(file.split(".pickle")[0]+"_metric.pickle"):
            with open(file.split(".pickle")[0]+"_metric.pickle", 'rb') as handle:
                rmse = pickle.load(handle)
                r2 = pickle.load(handle)
                f1 = pickle.load(handle)

            elbo = hist["val_decoder_loss"][idx]
            bacc = hist["val_decoder_acc"][idx]
            kl = hist["val_kl_loss"][idx]      
            cl = hist["val_classifier_loss"][idx]
            rl = hist["val_regressor_loss"][idx]
            
            # BRUCE ADD 20240111
            h5_file = file.split(".pickle")[0]+".h5"
            ENCODER, DECODER, MONITOR, IF_REG, IF_CLS, weights, LR, BS = get_spec(h5_file)

            model, pickle_file = train_vae(ENCODER, DECODER, MONITOR, IF_REG, IF_CLS,
                                           x_train, x_valid, y_train, y_valid, c_train, c_valid,
                                           l_train, l_valid, 1.0, weights, LR, BS, False) 
            
            latent_valid = latent_model(model, data=[x_valid, l_valid], enc_type=ENCODER, mean_var=True)
            
            z_mean = latent_valid[0]
            z_log_var = latent_valid[1]

            kl_total = - .5 * np.sum(1 + z_log_var -
                                            np.square(z_mean) -
                                            np.exp(z_log_var), axis=-1) * 1
            
            kl = kl_total.mean()


            if np.isnan(elbo).any() or np.isnan(bacc).any():
                continue
            else:
                elbos.append(elbo)
                baccs.append(bacc)
                kls.append(kl)
                cls.append(cl)
                rls.append(rl)
                rmses.append(rmse[1])
                r2s.append(r2[1])
                f1s.append(f1[1])
                files_select.append(file)
        else:
            continue
        
    elbos = np.array(elbos)
    baccs = np.array(baccs)
    kls = np.array(kls)
    cls = np.array(cls)
    rls = np.array(rls)
    rec = elbos-kls
    rmses = np.array(rmses)
    r2s = np.array(r2s)
    f1s = np.array(f1s)
    
    return elbos, baccs, kls, cls, rls, rec, rmses, r2s, f1s, files_select


def minmax(arr, mins=None, maxs=None):
    """
    Perform min-max scaling on an array.

    Args:
        arr (numpy.ndarray): The input array to be scaled.
        mins (float, optional): The custom minimum value for scaling. Default is None.
        maxs (float, optional): The custom maximum value for scaling. Default is None.

    Returns:
        numpy.ndarray: The scaled array.
    """
    # Determine minimum and maximum values for scaling
    if mins is not None:
        min_val = mins
    else:
        min_val = np.min(arr)

    if maxs is not None:
        max_val = maxs
    else:
        max_val = np.max(arr)

    # Perform min-max scaling
    scaled_arr = [(x - min_val) / (max_val - min_val) for x in arr]

    return scaled_arr


def pareto_frontier(baccs, r2s, f1s, kls, limits):
    """
    Find the Pareto frontier from a set of data points with multiple objectives.

    Args:
        baccs (list): List of balanced accuracy values.
        r2s (list): List of R2 values.
        f1s (list): List of F1 scores.
        kls (list): List of KL divergence values.
        limits (list): List of limit values for each objective.

    Returns:
        tuple: A tuple containing two elements:
            - pareto_indices (list): List of indices of Pareto points.
            - pareto_front (numpy.ndarray): Array of Pareto points satisfying the specified limits.
    """
    combined = list(zip(baccs, r2s, f1s, kls, range(len(baccs))))
    pareto_front = []

    for point in combined:
        is_dominated = False

        for other_point in combined:
            if all(other <= point_dim for other, point_dim in zip(other_point[:4], point[:4])) and any(other < point_dim for other, point_dim in zip(other_point[:4], point[:4])):
                is_dominated = True
                break

        if not is_dominated:
            pareto_front.append(point)

    pareto_indices = [
        point[4]
        for point in pareto_front
        if all(point_dim < limit for point_dim, limit in zip(point[:4], limits))
    ]

    pareto_front = [
        point[:5]
        for point in pareto_front
        if all(point_dim < limit for point_dim, limit in zip(point[:4], limits))
    ]

    return pareto_indices, np.array(pareto_front)


def closest_to_origin(pareto_front):
    """
    Find the point in the Pareto front closest to the origin in multi-dimensional space.

    Args:
        pareto_front (list): List of points in the Pareto front, each represented as a tuple of objectives.

    Returns:
        tuple: A tuple containing two elements:
            - closest_point (tuple): The point in the Pareto front closest to the origin.
            - closest_idx (int): The index of the closest point in the Pareto front.
    """
    baccs, r2s, f1s, kls, idx = zip(*pareto_front)

    baccs = minmax(baccs)
    r2s = minmax(r2s)
    f1s = minmax(f1s)
    kls = minmax(kls)
    
    scaled_pareto_front = list(zip(baccs, r2s, f1s, kls, idx))
    closest_tuple = min(scaled_pareto_front, key=lambda point: np.linalg.norm(np.array(point[:-1])))
    
    closest_point = closest_tuple[:-1]
    closest_idx = closest_tuple[-1]
    
    return closest_point, int(closest_idx)