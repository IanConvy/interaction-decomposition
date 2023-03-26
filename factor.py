import tensorflow as tf
import numpy as np

import models

def factorize(weights_dir, test_data):

    # This function "factors" (performs an interaction decomposition)
    # on the model saved in the weights_dir on the passed test_data.
    # The algorithm for the decomposition is specified in the model 
    # class.

    model = models.load_model(weights_dir)
    model.set_output(True, False)
    (x_test, y_test) = test_data
    num_samples = tf.shape(x_test)[0]
    batch_size = 128
    i = 0
    results = []
    while i*batch_size < num_samples:
        start = i*batch_size
        end = start + batch_size
        batch = x_test[start:end]
        factor_output = model(batch)
        results.append(factor_output)
        i += 1
        print(f"\r{start}", end = "")
    results = tf.concat(results, 0)
    np.savez_compressed(weights_dir + "/factors.npz",
        data = x_test.numpy(), labels = y_test.numpy(), results = results.numpy())    
