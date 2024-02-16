import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import ast
import pickle
import numpy as np
import tensorflow as tf

from timeit import default_timer as timer
from tensorflow.keras import layers, callbacks
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from spektral.layers import GINConvBatch

LATENT_DIM = 8
WEIGHT_DIR = '/scratch/gpfs/sj0161/topo_result/' 

class Sampling(layers.Layer):
    def __init__(self, seed=None, **kwargs):
        super(Sampling, self).__init__(**kwargs)
        self.seed = seed
        
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim), seed=self.seed)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class KLDivergenceLayer(layers.Layer):
    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs, beta=1.0):
        z_mean, z_log_var = inputs
        kl_batch = - .5 * K.sum(1 + z_log_var -
                                K.square(z_mean) -
                                K.exp(z_log_var), axis=-1) * beta
        kl_loss = K.mean(kl_batch)
        self.add_loss(kl_loss, inputs=inputs)
        self.add_metric(kl_loss, aggregation='mean', name='kl_loss')
        return inputs

def encoder_gnn(input_shape=(100, 100)):
    """
    Constructs a GNN encoder with the specified input shape.

    Args:
        input_shape (tuple, optional): Shape of the input adjacency matrix and feature matrix. 
                                       Defaults to (100, 100).

    Returns:
        tuple: A tuple containing the input tensors and the output tensor of the encoder.
    """
    A = tf.keras.Input(shape=input_shape, sparse=False, name='a_1')
    F = tf.keras.Input(shape=input_shape, name='f_1')

    x = GINConvBatch(32, activation='relu')([F, A])
    x = GINConvBatch(32, activation='relu')([x, A])

    x = layers.Flatten(name='flatten')(x)
    x_out = layers.Dense(32, activation='relu')(x)

    return [A, F], x_out


def encoder_desc_dnn(input_shape=(11,)):
    """
    Constructs a DNN encoder for descriptor data with the specified input shape.

    Args:
        input_shape (tuple, optional): The shape of the input descriptor data. Defaults to (11,).

    Returns:
        tuple: A tuple containing the input tensor and the output tensor of the encoder.
    """
    x_in = tf.keras.Input(shape=input_shape, name='x_in')

    x = layers.Dense(32, activation='relu')(x_in)
    x_out = layers.Dense(32, activation='relu')(x)

    return x_in, x_out


def encoder_desc_gnn(input_shape=[(100, 100), (11,)]):
    """
    Constructs a GNN-based encoder with an additional DNN branch for descriptor data.

    Args:
        input_shape (list of tuple, optional): List containing the shapes of the adjacency 
                                               matrix, feature matrix, and descriptor data. 
                                               Defaults to [(100, 100), (11,)].

    Returns:
        tuple: A tuple containing the input tensors and the concatenated output tensor 
               from the GNN and DNN branches of the encoder.
    """
    A1 = tf.keras.Input(shape=input_shape[0], sparse=False, name='a_1')
    F1 = tf.keras.Input(shape=input_shape[0], name='f_1')
    x_in = tf.keras.Input(shape=input_shape[1], name='x_in')
    
    x1 = GINConvBatch(32, activation='relu')([F1, A1])
    x1 = GINConvBatch(32, activation='relu')([x1, A1])
    
    x2 = layers.Dense(32, activation='relu')(x_in)
    
    x_out2 = layers.Dense(32, activation='relu')(x2)
    x1 = layers.Flatten(name='flatten')(x1)
    x_out1 = layers.Dense(32, activation='relu')(x1)
    
    x_out = layers.Concatenate()([x_out1, x_out2])

    return [[A1, F1], x_in], x_out

def latent_space(x_out, beta=1.0):
    """
    Constructs the latent space for a VAE, including the KL divergence layer.

    Args:
        x_out (tf.Tensor): Output tensor from the encoder.
        beta (float, optional): Weight for the KL divergence term in the loss function. Defaults to 1.0.

    Returns:
        tuple: Tensors representing the mean and log variance of the latent space.
    """
    z_mean = layers.Dense(LATENT_DIM, name="z1")(x_out)
    z_log_var = layers.Dense(LATENT_DIM, name="z2")(x_out)
    z_mean, z_log_var = KLDivergenceLayer(name='kl')([z_mean, z_log_var], beta=beta)
    return z_mean, z_log_var

def regressor_dnn():
    """
    Constructs a dense neural network (DNN) based regressor model.

    Returns:
        tf.keras.Model: A Keras model for regression, mapping from the latent space to a single output.
    """
    r_in = layers.Input(shape=(LATENT_DIM,))
    x = layers.Dense(32, activation='relu', name="r1")(r_in)
    x = layers.Dense(32, activation='relu', name="r2")(x)
    r_out = layers.Dense(1, activation='linear', name="r3")(x)
    regressor = tf.keras.Model(r_in, r_out, name='regressor')
    return regressor


def classifier_dnn():
    """
    Constructs a dense neural network (DNN) based classifier model.

    Returns:
        tf.keras.Model: A Keras model for classification, mapping from the latent space to a softmax output.
    """
    c_in = layers.Input(shape=(LATENT_DIM,))
    x = layers.Dense(32, activation='relu', name="c1")(c_in)
    x = layers.Dense(32, activation='relu', name="c2")(x)
    c_out = layers.Dense(6, activation='softmax', name="c3")(x)
    classifier = tf.keras.Model(c_in, c_out, name='classifier')
    return classifier


def decoder_cnn():
    """
    Constructs a convolutional neural network (CNN) based decoder model.

    Returns:
        tf.keras.Model: A Keras model for decoding, mapping from the latent space to the reconstructed output.
    """
    d_in = layers.Input(shape=(LATENT_DIM,))
    x = layers.Dense(32, activation='relu', name="d1")(d_in)
    x = layers.Dense(32, activation='relu', name="d2")(x)
    x = layers.Dense(25 * 25 * 64, activation='relu', name="d3")(x)
    x = layers.Reshape((25, 25, 64), name="d4")(x)
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, padding='same', activation='relu', name=f"d5")(x)
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, padding='same', activation='relu', name=f"d6")(x)
    x = layers.Conv2DTranspose(1, (3, 3), padding='same', activation='sigmoid', name="d7")(x)
    d_out = layers.Reshape((100, 100), name="d8")(x)
    decoder = tf.keras.Model(d_in, d_out, name='decoder')
    return decoder


def get_callbacks(weight_path, monitor='val_decoder_acc', patience=200):
    """
    Generates model training callbacks.

    Args:
        weight_path (str): Path to save model weights.
        monitor (str): Metric to monitor for performance (default 'val_decoder_acc').
        patience (int): Number of epochs to wait for improvement before stopping (default 200).

    Returns:
        list: ModelCheckpoint and EarlyStopping callbacks.
    """
    mode = 'max' if 'acc' in monitor else 'min'
    checkpoint = callbacks.ModelCheckpoint(weight_path, monitor=monitor, mode=mode, 
                                           save_weights_only=True, save_best_only=True, verbose=0)
    early_stop = callbacks.EarlyStopping(monitor=monitor, mode=mode, patience=patience)
    
    return [checkpoint, early_stop]


def get_model(beta=1.0, enc_type='gnn', dec_type='cnn', if_reg=True, if_cls=True, seed=42):
    """
    Constructs and returns a VAE model based on the specified encoder 
    and decoder types, with optional regression and classification tasks.

    Args:
        beta (float, optional): The beta parameter for the VAE KL term Default is 1.0.
        enc_type (str, optional): The type of encoder to use. Default is 'gnn'.
        dec_type (str, optional): The type of decoder to use. Default is 'cnn'.
        if_reg (bool, optional): If a regression tasks should be added. Default is True.
        if_cls (bool, optional): If a classification task should be added. Default is True.
        seed (int, optional): Seed for the random number generator in the sampling layer. Default is 42.

    Returns:
        tf.keras.Model: The constructed VAE model with the specified configuration. 
    """
        
    if enc_type == 'gnn':
        x_in, x_out = encoder_gnn()
        
    elif enc_type == 'desc_dnn':
        x_in, x_out = encoder_desc_dnn()
        
    elif enc_type == 'desc_gnn':
        x_in, x_out = encoder_desc_gnn()
        
    else:
        raise Exception("Option not available.")

    z_mean, z_log_var = latent_space(x_out, beta=beta)
    z = Sampling(seed=seed)([z_mean, z_log_var])
    
    if dec_type == 'cnn':
        dec = decoder_cnn()
    else:
        raise Exception("Option not available.")
        
    d_out = dec(z)
    
    if if_reg:
        reg = regressor_dnn()
        r_out = reg(z)

    if if_cls:
        cls = classifier_dnn()
        c_out = cls(z)
    
    if if_reg and if_cls:
        model = tf.keras.Model(inputs=x_in, outputs=[d_out, r_out, c_out])
        
    elif if_reg and not if_cls:
        model = tf.keras.Model(inputs=x_in, outputs=[d_out, r_out])
        
    elif not if_reg and if_cls:
        model = tf.keras.Model(inputs=x_in, outputs=[d_out, c_out])
        
    elif not if_reg and not if_cls:
        model = tf.keras.Model(inputs=x_in, outputs=[d_out])
        
    return model

def get_spec(file):
    """
    Extract specifications from a filename.

    Args:
        file (str): The file name from which specifications are to be extracted.

    Returns:
        tuple: A tuple containing the following extracted specifications:
               - ENCODER (str): The encoder specification.
               - DECODER (str): The decoder specification.
               - MONITOR (str): The monitor specification.
               - IF_REG (bool): Flag indicating if regression is used.
               - IF_CLS (bool): Flag indicating if classification is used.
               - weights (dict): Weights for loss functions.
               - LR (float): Learning rate.
               - BS (int): Batch size.
    """
    root = file.split("/")[-1].split("_")
    if "desc" in root:
        ENCODER = "_".join(root[:2])
        root2 = root[2:]
    else:
        ENCODER = root[0]
        root2 = root[1:]

    DECODER = root2[0]

    if len(root2) == 12:
        MONITOR = "_".join(root2[3:6])
        root3 = root2[6:]
    else:
        MONITOR = "_".join(root2[3:5])
        root3 = root2[5:]

    IF_REG = root3[0] == "True"
    IF_CLS = root3[1] == "True"

    weights = ast.literal_eval(root3[3])

    LR = float(root3[4])
    BS = int(root3[5].split(".h5")[0])
    return ENCODER, DECODER, MONITOR, IF_REG, IF_CLS, weights, LR, BS



def train_vae(enc_type, dec_type, monitor, if_reg, if_cls, x_train, x_valid, y_train, y_valid,
              c_train, c_valid, l_train, l_valid, beta=1.0, weights=[1.0, 1.0, 1.0],
              lr=0.001, bs=32, if_train=False, n_class=6, n_epoch=1000, date='20230828'):
    """
    This function trains a VAE model. It handles different types of encoders and decoders, 
    allows for regression and classification tasks, and accommodates various training configurations. 
    The function can also load pre-trained weights instead of training from scratch.

    Args:
        enc_type (str): The type of encoder to use.
        dec_type (str): The type of decoder to use.
        monitor (str): The parameter to monitor during training.
        if_reg (bool): Flag indicating if regression is to be performed.
        if_cls (bool): Flag indicating if classification is to be performed.
        x_train (numpy.ndarray): Training graph features.
        x_valid (numpy.ndarray): Validation graph features.
        y_train (numpy.ndarray): Training rg2.
        y_valid (numpy.ndarray): Validation rg2.
        c_train (numpy.ndarray): Training topology class.
        c_valid (numpy.ndarray): Validation topology class.
        l_train (numpy.ndarray): Training topological descriptors
        l_valid (numpy.ndarray): Validation topological descriptors
        beta (float, optional): The beta parameter for VAE. Default is 1.0.
        weights (list, optional): List of weights for different components of the loss function. Default is [1.0, 1.0, 1.0].
        lr (float, optional): Learning rate. Default is 0.001.
        bs (int, optional): Batch size. Default is 32.
        if_train (bool, optional): Flag indicating if the model should be trained or loaded. Default is False.
        n_class (int, optional): Number of classes for classification. Default is 6.
        n_epoch (int, optional): Number of epochs for training. Default is 1000.
        date (str, optional): Date string used for file naming. Default is '20230828'.

    Returns:
        tuple: A tuple containing the trained or loaded model and the path to the history file.
    """
    
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
    
    model = get_model(beta=beta, enc_type=enc_type, dec_type=dec_type, if_reg=if_reg, if_cls=if_cls)
    model, loss_weights = compile_model(model, lr=lr, if_reg=if_reg, if_cls=if_cls, weights=weights)
    
    weight_name, hist_name = get_file_names(enc_type, dec_type, date, LATENT_DIM, monitor, 
                                            if_reg, if_cls, beta, loss_weights, lr, bs)
    
    
    if (not if_reg) and (not if_cls) and ('acc' in monitor):
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
            train_label.append(to_categorical(c_train, n_class))
            valid_label.append(to_categorical(c_valid, n_class))
            
        hist = model.fit(in_train, train_label, validation_data=(in_valid, valid_label),
                         callbacks=[c1, c2], epochs=n_epoch, verbose=0, batch_size=bs)
        
        with open(os.path.join(WEIGHT_DIR, hist_name), 'wb') as handle:
            pickle.dump(hist.history, handle)
    else:
        model.load_weights(os.path.join(WEIGHT_DIR, weight_name))
        
    t1 = timer()
    
    print(weight_name + f' finished in {t1-t0:0.2f} sec ...')

    return model, os.path.join(WEIGHT_DIR, hist_name)


def rec_loss(y_true, y_pred):
    """
    Reconstruction loss for a binary classification problem.

    Args:
        y_true (tf.Tensor): True labels.
        y_pred (tf.Tensor): Predicted labels.

    Returns:
        tf.Tensor: Binary cross-entropy loss multiplied by 10000.
    """
    y_true_ = K.reshape(y_true, (-1, 100 * 100))
    y_pred_ = K.reshape(y_pred, (-1, 100 * 100))
    loss = tf.keras.losses.binary_crossentropy(y_true_, y_pred_) * 10000
    return loss

def reg_loss(y_true, y_pred):
    """
    Regression loss using mean absolute error.

    Args:
        y_true (tf.Tensor): True values.
        y_pred (tf.Tensor): Predicted values.

    Returns:
        tf.Tensor: Mean absolute error loss.
    """
    return tf.keras.losses.mean_absolute_error(y_true, y_pred)

def cls_loss(y_true, y_pred):
    """
    Classification loss using categorical cross-entropy.

    Args:
        y_true (tf.Tensor): True class labels (one-hot encoded).
        y_pred (tf.Tensor): Predicted class probabilities.

    Returns:
        tf.Tensor: Categorical cross-entropy loss.
    """
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred)

def acc(y_true, y_pred):
    """
    Compute balanced accuracy metric for binary classification.

    Balanced accuracy is a metric that takes into account both sensitivity (true positive rate)
    and specificity (true negative rate) to provide a balanced measure of classification performance.

    Args:
        y_true (tf.Tensor): True labels (ground truth).
        y_pred (tf.Tensor): Predicted labels (probabilities or binary predictions).

    Returns:
        tf.Tensor: Balanced accuracy score.
    """
    y_true_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred)
    y_pred_bin = K.round(y_pred_flat)
    TP = K.sum(y_true_flat * y_pred_bin)
    FP = K.sum((1-y_true_flat) * y_pred_bin)
    TN = K.sum((1-y_true_flat) * (1-y_pred_bin))
    FN = K.sum(y_true_flat * (1-y_pred_bin))
    sensitivity = TP / (TP + FN + K.epsilon())
    specificity = TN / (TN + FP + K.epsilon())
    balanced_accuracy = (sensitivity + specificity) / 2
    return balanced_accuracy


def get_file_names(enc_type, dec_type, date, dim, monitor, if_reg, if_cls, beta, loss_weights, lr, bs):
    """
    Generate file names based on various parameters.

    Args:
        enc_type (str): The encoding type.
        dec_type (str): The decoding type.
        date (str): The date or timestamp.
        dim (int): The dimensionality.
        monitor (str): The monitoring type.
        if_reg (bool): Whether regularization is used.
        if_cls (bool): Whether classification is used.
        beta (float): The beta value.
        loss_weights (list): List of loss weights.
        lr (float): Learning rate.
        bs (int): Batch size.

    Returns:
        tuple of str: A tuple containing two file names in the format (model_file_name, pickle_file_name).
    """
    base_name = f"{enc_type}_{dec_type}_{date}_{dim}_{monitor}_{if_reg}_{if_cls}_{beta}_{loss_weights}_{lr}_{bs}"
    return base_name + ".h5", base_name + ".pickle"



def compile_model(model, lr=0.001, if_reg=True, if_cls=True, weights=[1.0, 1.0, 1.0]):
    """
    Compile a Keras model with specified configuration.

    Args:
        model (tf.keras.Model): The Keras model to compile.
        lr (float, optional): Learning rate for the optimizer. Default is 0.001.
        if_reg (bool, optional): Whether to include a regularization loss. Default is True.
        if_cls (bool, optional): Whether to include a classification loss. Default is True.
        weights (list of float, optional): Loss weights for different components.
            Default is [1.0, 1.0, 1.0], and it should have at least 2 elements.

    Returns:
        tf.keras.Model: Compiled Keras model.
        list of float: Loss weights used during compilation.
    """
    loss = [rec_loss]
    loss_weights = [weights[0]]
    if if_reg:
        loss.append(reg_loss)
        loss_weights.append(weights[1])
    if if_cls:
        loss.append(cls_loss)
        try:
            loss_weights.append(weights[2])
        except:
            loss_weights.append(weights[1])
    
    if not if_reg and not if_cls:
        model.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=lr),
            loss=loss,
            metrics={'decoder': acc}
        )
    else:
        model.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=lr),
            loss=loss,
            loss_weights=loss_weights, 
            metrics={'decoder': acc}
        )
    return model, loss_weights


def latent_model(model, data, enc_type='gnn', mean_var=False):
    """
    Generates latent space representations from a given model and data.

    Args:
        model (tf.keras.Model): The model from which to generate the latent representations.
        data (list or numpy.ndarray): The input data for the model. This can vary based on encoder type.
        enc_type (str, optional): The type of encoder used in the model. Defaults to 'gnn'.
        mean_var (bool, optional): If True, returns both the mean and variance from the latent space. 
                                   Defaults to False.

    Returns:
        numpy.ndarray or tuple: The latent space representation or a tuple of mean and variance representations.
    """
    # Prepare input based on encoder type
    if enc_type == 'gnn':
        x_in = [data[0], data[0]]
    elif enc_type == 'desc_gnn':
        x_in = [[data[0], data[0]], data[1]]
    elif enc_type == 'desc_dnn':
        x_in = data[1]
    else:
        x_in = data[0]

    # Generate latent space representations
    if mean_var:
        l1_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('z1').output)
        l1 = l1_model.predict(x_in, verbose=0)
        l2_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('z2').output)
        l2 = l2_model.predict(x_in, verbose=0)
        return l1, l2
    else:
        l_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('sampling').output)
        l = l_model.predict(x_in, verbose=0)
        return l

