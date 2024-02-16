import tensorflow as tf

from tensorflow.keras.utils import to_categorical

def compute_saliency(model, x_train, y_train, l_train, c_train, output_index, enc_type, if_reg, if_cls):
    """
    Computes the saliency map for the given model and data.

    Args:
        model (tf.keras.Model): The model for which to compute the saliency.
        x_train (numpy.ndarray): Input feature data.
        y_train (numpy.ndarray): Regression target data.
        l_train (numpy.ndarray): Additional input data for descriptor models.
        c_train (numpy.ndarray): Classification target data.
        output_index (int): The index of the output to use for computing saliency.
        enc_type (str): The type of encoder used in the model ('desc_gnn' or 'desc_dnn').
        if_reg (bool): Flag indicating if the model includes regression.
        if_cls (bool): Flag indicating if the model includes classification.

    Returns:
        numpy.ndarray or None: The computed saliency map as a NumPy array, or None if regression is not included.
    """
    if if_reg:
        # Prepare input data based on encoder type
        if enc_type == 'desc_gnn':
            input_data = [[tf.convert_to_tensor(x_train), tf.convert_to_tensor(x_train)], 
                          tf.convert_to_tensor(l_train)]
        elif enc_type == 'desc_dnn':
            input_data = tf.convert_to_tensor(l_train)
        
        # Prepare target data based on classification flag
        if if_cls:
            target_data = [tf.convert_to_tensor(x_train), 
                           tf.convert_to_tensor(y_train), 
                           tf.convert_to_tensor(to_categorical(c_train, 6))]
        else:
            target_data = [tf.convert_to_tensor(x_train), 
                           tf.convert_to_tensor(y_train)]

        # Compute gradients for saliency
        with tf.GradientTape() as tape:
            tape.watch(input_data)
            output = model(input_data)
            loss = model.loss[output_index](target_data[output_index], output[output_index])

        # Get gradients for the relevant input
        if enc_type == 'desc_gnn':
            grads = tape.gradient(loss, input_data[1])
        elif enc_type == 'desc_dnn':
            grads = tape.gradient(loss, input_data)

        # Normalize gradients
        grads = tf.abs(grads)
        grads = (grads - tf.reduce_min(grads)) / (tf.reduce_max(grads) - tf.reduce_min(grads))

        return grads.numpy()
    else:
        return None