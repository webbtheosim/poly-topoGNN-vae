import os

import numpy as np
import networkx as nx
import tensorflow as tf

from graph_utils import graph_anneal_break_largest_circle
from data_utils import load_data, get_desc

WEIGHT_DIR = '/scratch/gpfs/sj0161/topo_result/' # change to your directory
DATA_DIR = '/scratch/gpfs/sj0161/topo_data/' # change to your directory

LATENT_DIM = 8


def reg_cls(model, z, data, enc_type='gnn'):
    """
    Perform regression and classification using a given model.

    Args:
        model (tf.keras.Model): The neural network model.
        z (numpy.ndarray): Input data for regression and classification.
        data (numpy.ndarray or list): Input data for encoding (depends on 'enc_type' argument).
        enc_type (str, optional): The encoding type, either 'gnn', 'desc_gnn', or other. Defaults to 'gnn'.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]: A tuple containing:
        - generated_reg (numpy.ndarray): Predictions from the regression model.
        - generated_cls (numpy.ndarray): Predictions from the classification model.
        - cleaned_reg (numpy.ndarray): Predictions after cleaning from the regression model.
        - cleaned_cls (numpy.ndarray): Predictions after cleaning from the classification model.
    """
    # Regression Model
    regressor = model.get_layer('regressor')
    r_in = tf.keras.Input(shape=(LATENT_DIM,), name='reg_in')
    for i, l in enumerate(regressor.layers):
        if i == 0:
            x = l(r_in)
        else:
            x = l(x)
    reg_model = tf.keras.Model(inputs=r_in, outputs=x)
    generated_reg = reg_model.predict(z, verbose=0)

    # Classification Model
    classifier = model.get_layer('classifier')
    c_in = tf.keras.Input(shape=(LATENT_DIM,), name='cls_in')
    for i, l in enumerate(classifier.layers):
        if i == 0:
            x = l(c_in)
        else:
            x = l(x)
    cls_model = tf.keras.Model(inputs=c_in, outputs=x)
    generated_cls = cls_model.predict(z, verbose=0)

    # Encoding Type Handling
    if enc_type == 'gnn':
        _, cleaned_reg, cleaned_cls = model.predict([data, data], verbose=0)
    elif enc_type == 'desc_gnn':
        _, cleaned_reg, cleaned_cls = model.predict([[data[0], data[0]], data[1]], verbose=0)
    else:
        _, cleaned_reg, cleaned_cls = model.predict(data, verbose=0)
        
    return generated_reg, generated_cls, cleaned_reg, cleaned_cls

def polymer_generation(model, latent_mean, latent_log_var, enc_type='gnn'):
    """
    Generate and clean a polymer graph using a given model.

    Args:
        model (tf.keras.Model): The neural network model.
        latent_mean (numpy.ndarray): Mean of the latent space.
        latent_log_var (numpy.ndarray): Log variance of the latent space.
        enc_type (str, optional): Encoding type: 'gnn', 'desc_dnn', or 'desc_gnn'. Defaults to 'gnn'.

    Returns:
        Tuple[networkx.Graph, networkx.Graph, str, str, float, float, float]:
            - G_pre_clean (networkx.Graph): Generated polymer graph before cleaning.
            - G_post_clean (networkx.Graph): Generated polymer graph after cleaning.
            - cls_pre_clean (str): Predicted class before cleaning.
            - cls_post_clean (str): Predicted class after cleaning.
            - reg_pre_clean (float): Regression result before cleaning.
            - reg_post_clean_m (float): Mean of regression result after cleaning.
            - reg_post_clean_s (float): Standard deviation of regression result after cleaning.
    """
    
    (_, _, _, NAMES, SCALER, _) = load_data(os.path.join(DATA_DIR, 'rg2.pickle'), fold=0, if_validation=True)

    # Get decoder weights
    decoder = model.get_layer('decoder')

    # Define input tensor for generation
    d_in = tf.keras.Input(shape=(LATENT_DIM,), name='gen_d_in')

    # Pass the input through the decoder layers
    x = d_in
    for layer in decoder.layers:
        x = layer(x)
    
    # Define generation model
    gen_model = tf.keras.Model(inputs=d_in, outputs=x)

    # Prepare latent space
    if len(latent_mean.shape) == 1:
        latent_mean = latent_mean[None, ...]
    elif len(latent_mean.shape) == 2 and latent_mean.shape[0] != 1:
        raise Exception("Only allow one sample at a time ...")

    epsilon = np.random.normal(size=(1, LATENT_DIM))

    # Generate data
    if latent_log_var is None:
        z_sample = latent_mean
    else:
        z_sample = latent_mean + np.exp(0.5 * latent_log_var) * epsilon

    sampled_data = np.round(gen_model.predict(z_sample, verbose=0))[0]

    # Convert data to graphs
    G_pre_clean = nx.from_numpy_array(sampled_data)
    G_post_clean = graph_anneal_break_largest_circle(sampled_data)
    
    if enc_type == 'desc_dnn':
        data = get_desc(G_post_clean)[None, ]
        data = np.where(np.isnan(data), -2, data)
        data = SCALER.transform(data)

    elif enc_type == 'desc_gnn':
        adjs = np.zeros((1, 100, 100))
        adj_ = nx.to_numpy_array(G_post_clean)
        adjs[0, :len(adj_), :len(adj_)] = adj_
        data1 = adjs

        data2 = get_desc(G_post_clean)[None, ]
        data2 = np.where(np.isnan(data2), -2, data2)
        data2 = SCALER.transform(data2)

        data = [data1, data2]

    else:
        adjs = np.zeros((1, 100, 100))
        adj_ = nx.to_numpy_array(G_post_clean)
        adjs[0, :len(adj_), :len(adj_)] = adj_
        data = adjs

    (reg_pre_clean, cls_pre_clean, reg_post_clean, cls_post_clean) = reg_cls(model, z_sample, data, enc_type)

    cls_pre_clean = NAMES[np.argmax(cls_pre_clean, axis=1)]
    cls_post_clean = NAMES[np.argmax(cls_post_clean, axis=1)]

    unique_cls, counts = np.unique(cls_pre_clean, return_counts=True)
    cls_pre_clean = unique_cls[np.argmax(counts)]

    unique_cls, counts = np.unique(cls_post_clean, return_counts=True)
    cls_post_clean = unique_cls[np.argmax(counts)]

    reg_pre_clean = reg_pre_clean[0][0]

    reg_post_clean_m = np.mean(reg_post_clean)
    reg_post_clean_s = np.std(reg_post_clean)
    
    return G_pre_clean, G_post_clean, cls_pre_clean, cls_post_clean, reg_pre_clean, reg_post_clean_m, reg_post_clean_s


def check_valid(gen_reg, cln_reg_m, gen_cls, cln_cls, threshold=2.0):
    """
    Check the validity of generated and cleaned data based on regression and classification results.

    Args:
        gen_reg (float): Regression result from generated data.
        cln_reg_m (float): Mean of regression result from cleaned data.
        gen_cls (str): Classification result from generated data.
        cln_cls (str): Classification result from cleaned data.
        threshold (float, optional): Threshold for comparing regression results. Defaults to 2.0.

    Returns:
        bool: True if both regression and classification results are valid, False otherwise.
    """
    flag1 = np.abs(gen_reg - cln_reg_m) < threshold
    flag2 = gen_cls == cln_cls
    return flag1 and flag2