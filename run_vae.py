import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pickle
import itertools
import numpy as np

from timeit import default_timer as timer

from graph_utils import *
from data_utils import *
from model_utils import *
from analysis_utils import *
from saliency_utils import *
from generation_utils import *

DATA_DIR = '/scratch/gpfs/sj0161/topo_data/'
WEIGHT_DIR = '/scratch/gpfs/sj0161/topo_result/'
ANALYSIS_DIR = '/scratch/gpfs/sj0161/topo_analysis/'


def train_vae(enc_type, dec_type, monitor,
              if_reg, if_cls,
              x_train, x_valid,
              y_train, y_valid,
              c_train, c_valid,
              l_train, l_valid,
              beta=1.0, weights=[1.0, 1.0, 1.0],
              lr=0.001, bs=32,
              if_train=False):
    K.clear_session()
    t0 = timer()
    
    if enc_type == 'desc_dnn':
        in_train = np.copy(l_train)
        in_valid = np.copy(l_valid)
    elif enc_type == 'desc_gnn':
        in_train = [[np.copy(x_train), np.copy(x_train)], np.copy(l_train)]
        in_valid = [[np.copy(x_valid), np.copy(x_valid)], np.copy(l_valid)]
    elif enc_type == 'gnn':
        in_train = [np.copy(x_train), np.copy(x_train)]
        in_valid = [np.copy(x_valid), np.copy(x_valid)]
    else:
        in_train = np.copy(x_train)
        in_valid = np.copy(x_valid)
    
    model = get_model(beta=beta, enc_type=enc_type, dec_type=dec_type,
                      if_reg=if_reg, if_cls=if_cls)
    
    model, loss_weights = compile_model(model, lr=lr, if_reg=if_reg, if_cls=if_cls, weights=weights)
    
    weight_name, hist_name = get_file_names(enc_type, dec_type, "20230828", 
                                            LATENT_DIM, monitor, 
                                            if_reg, if_cls, beta, loss_weights, lr, bs)
    
    
    print(weight_name + ' started ...')
    
    if not if_reg and not if_cls and 'acc' in monitor:
        monitor = 'val_acc'
    
    if if_train:
        c1, c2 = get_callbacks(os.path.join(WEIGHT_DIR, weight_name), monitor=monitor)
        
        if not if_reg and not if_cls:
            train_label = x_train
            valid_label = x_valid
        else:
            train_label = [x_train]
            valid_label = [x_valid]
        
        if if_reg:
            train_label.append(y_train)
            valid_label.append(y_valid)
            
        if if_cls:
            train_label.append(to_categorical(c_train, 6))
            valid_label.append(to_categorical(c_valid, 6))
            
        hist = model.fit(
            in_train, train_label, validation_data=(in_valid, valid_label),
            callbacks=[c1, c2], epochs=1000, verbose=0, batch_size=bs)
        
        with open(os.path.join(WEIGHT_DIR, hist_name), 'wb') as handle:
            pickle.dump(hist.history, handle)
    else:
        model.load_weights(os.path.join(WEIGHT_DIR, weight_name))
        
    t1 = timer()
    
    print(weight_name + f' finished in {t1-t0:0.2f} sec ...')

    return model, os.path.join(WEIGHT_DIR, hist_name)

if __name__ == '__main__':
    (x_train, y_train, c_train, l_train, graph_train), \
    (x_valid, y_valid, c_valid, l_valid, graph_valid), \
    (x_test, y_test, c_test, l_test, graph_test), \
    NAMES, SCALER = load_data(fold=0, if_validation=True)

    has_nan = np.isnan(l_train).any()
    has_inf = np.isinf(l_train).any()

    print("l_train contains NaN values:", has_nan)
    print("l_train contains inf values:", has_inf)

    K.clear_session()

    LATENT_DIM = 8
    DECODER = "cnn"

    idx_slurm = int(os.environ["SLURM_ARRAY_TASK_ID"])

    encs = ["desc_gnn", "gnn", "desc_dnn"]
    mons = ["val_decoder_loss", "val_decoder_acc", "val_loss"]

    encmons = list(itertools.product(encs, mons)) # shape 9

    ENCODER = encmons[idx_slurm][0]
    MONITOR = encmons[idx_slurm][1]

    for IF_REG in [False, True]:
        for IF_CLS in [False, True]:
            if IF_REG == False or IF_CLS == False:
                for LR in [1e-4, 1e-3, 1e-2]:
                    for BS in [32, 64, 128]:
                        for rw in [0.01, 0.1, 1, 10, 100]:
                            for cw in [0.01, 0.1, 1, 10, 100]:
                                if IF_REG == False and IF_CLS == False:
                                    if MONITOR == "val_decoder_acc":
                                        MONITOR = "val_acc"
                                    elif MONITOR == "val_decoder_loss":
                                        MONITOR = "val_loss"

                                weight_name, hist_name = get_file_names(ENCODER, DECODER, "20230829", 
                                                                        LATENT_DIM, MONITOR, 
                                                                        IF_REG, IF_CLS, 1.0, [1.0, rw, cw], LR, BS)



                                if os.path.exists(os.path.join(WEIGHT_DIR, hist_name)):
                                    if_train = False
                                else:
                                    if_train = True

                                model, pickle_file = train_vae(ENCODER, DECODER, MONITOR,
                                                IF_REG, IF_CLS,
                                                x_train, x_valid,
                                                y_train, y_valid,
                                                c_train, c_valid,
                                                l_train, l_valid,
                                                1.0, [1.0, rw, cw],
                                                LR, BS,
                                                if_train)

                                with open(pickle_file, 'rb') as handle:
                                    hist = pickle.load(handle)