import tensorflow as tf

from src import data

# This module builds and trains an Inception-like CNN to serve as a comparison
# to the tensor network models.

def conv2D_bn_relu(x, filters, kernel_size, strides, padding = 'valid', kernel_initializer = 'glorot_uniform', name = None):
    
    # This function builds a 2D convolution with batch normalization and ReLU activation.

    x = tf.keras.layers.Conv2D(filters = filters, 
                      kernel_size = kernel_size, 
                      strides = strides, 
                      padding = padding, 
                      kernel_initializer = kernel_initializer,
                      name = name,
                      use_bias = False)(x)
    x = tf.keras.layers.BatchNormalization(scale = False)(x)
    a = tf.keras.layers.Activation('relu')(x)
    return a


def inception_module_A(x, filters = None, kernel_initializer = 'glorot_uniform'):
    
    # This function builds the Inception module A as described in Figure 4 
    # of "Inception-v4, Inception-ResNet and the Impact of Residual Connections 
    # on Learning" (Szegedy, et al. 2016).
    
    # Arguments
    #   x: 4D tensor with shape: `(batch, rows, cols, channels)`.
    #   filters: Number of output filters for the module.
    #   kernel_initializer: Weight initializer for all convolutional layers in module.
    
    if filters is None:
        filters = int(x.shape[-1])
    branch_filters = filters // 4
        
    b1 = conv2D_bn_relu(x, 
                        filters = (branch_filters // 3) * 2, 
                        kernel_size = 1, 
                        strides = 1, 
                        kernel_initializer = kernel_initializer)
    b1 = conv2D_bn_relu(b1, 
                        filters = branch_filters, 
                        kernel_size = 3, 
                        strides = 1, 
                        padding = 'same', 
                        kernel_initializer = kernel_initializer)
    
    b2 = conv2D_bn_relu(x, 
                        filters = (branch_filters // 3) * 2, 
                        kernel_size = 1, 
                        strides = 1, 
                        kernel_initializer = kernel_initializer)
    b2 = conv2D_bn_relu(b2, 
                        filters = branch_filters, 
                        kernel_size = 3, 
                        strides = 1, 
                        padding = 'same', 
                        kernel_initializer = kernel_initializer)
    b2 = conv2D_bn_relu(b2, 
                        filters = branch_filters, 
                        kernel_size = 3, 
                        strides = 1, 
                        padding = 'same', 
                        kernel_initializer = kernel_initializer)
        
    b3 = conv2D_bn_relu(x, 
                        filters = branch_filters, 
                        kernel_size = 1, 
                        strides = 1, 
                        kernel_initializer = kernel_initializer)
    
    pool = tf.keras.layers.AveragePooling2D(pool_size = (3, 3), strides = 1, padding = 'same')(x)
    pool = conv2D_bn_relu(pool, 
                          filters = branch_filters, 
                          kernel_size = 1, 
                          strides = 1, 
                          kernel_initializer = kernel_initializer)
    a = tf.keras.layers.concatenate([b1, b2, b3, pool])
    return a

def inception_module_C(x, filters=None, kernel_initializer = 'glorot_uniform'):
    
    # This function builds the Inception module C as described in Figure 6 
    # of "Inception-v4, Inception-ResNet and the Impact of Residual Connections 
    # on Learning" (Szegedy, et al. 2016).
    
    # Arguments
    #   x: 4D tensor with shape: `(batch, rows, cols, channels)`.
    #   filters: Number of output filters for the module.
    #   kernel_initializer: Weight initializer for all convolutional layers in module.
        
    if filters is None:
        filters = int(x.shape[-1])
    branch_filters = filters // 6
        
    b1 = conv2D_bn_relu(x, 
                        filters = (branch_filters // 2) * 3, 
                        kernel_size = 1, 
                        strides = 1, 
                        kernel_initializer = kernel_initializer)
        
    b1a = conv2D_bn_relu(b1, 
                         filters = branch_filters, 
                         kernel_size = (1, 3), 
                         strides = 1, 
                         padding = 'same', 
                         kernel_initializer = kernel_initializer)
    
    b1b = conv2D_bn_relu(b1, 
                         filters = branch_filters, 
                         kernel_size = (3, 1), 
                         strides = 1, 
                         padding = 'same', 
                         kernel_initializer = kernel_initializer)
    
    b2 = conv2D_bn_relu(x, 
                        filters = (branch_filters // 2) * 3, 
                        kernel_size = 1, 
                        strides = 1, 
                        kernel_initializer = kernel_initializer)
    b2 = conv2D_bn_relu(b2, 
                        filters = (branch_filters // 4) * 7, 
                        kernel_size = (1, 3), 
                        strides = 1, 
                        padding = 'same', 
                        kernel_initializer = kernel_initializer)
    b2 = conv2D_bn_relu(b2, 
                        filters = branch_filters * 2, 
                        kernel_size = (3, 1), 
                        strides = 1, 
                        padding = 'same', 
                        kernel_initializer = kernel_initializer)

    b2a = conv2D_bn_relu(b2, 
                         filters = branch_filters, 
                         kernel_size = (1, 3), 
                         strides = 1, 
                         padding = 'same', 
                         kernel_initializer = kernel_initializer)
    
    b2b = conv2D_bn_relu(b2, 
                         branch_filters, 
                         kernel_size = (3, 1), 
                         strides = 1, 
                         padding = 'same', 
                         kernel_initializer = kernel_initializer)
        
    b3 = conv2D_bn_relu(x, 
                        filters = branch_filters, 
                        kernel_size = 1, 
                        strides = 1, 
                        kernel_initializer = kernel_initializer)
    
    pool = tf.keras.layers.AveragePooling2D(pool_size = (2, 2), strides = 1, padding = 'same')(x)
    pool = conv2D_bn_relu(pool, 
                          filters = branch_filters, 
                          kernel_size = 1, 
                          strides = 1, 
                          kernel_initializer = kernel_initializer)
    a = tf.keras.layers.concatenate([b1a, b1b, b2a, b2b, b3, pool])
    return a

def reduction_module_A(x, filters, kernel_initializer = 'glorot_uniform'):
    
    # This function builds the reduction module A as described in Figure 7 
    # of "Inception-v4, Inception-ResNet and the Impact of Residual Connections 
    # on Learning" (Szegedy, et al. 2016).
    
    # Arguments
    #   x: 4D tensor with shape: `(batch, rows, cols, channels)`.
    #   filters: Number of output filters for the module.
    #   kernel_initializer: Weight initializer for all convolutional layers in module.
    
    branch_filters = (filters - int(x.shape[-1])) // 2
        
    b1 = conv2D_bn_relu(x, 
                        filters = branch_filters, 
                        kernel_size = 3, 
                        strides = 1, 
                        padding = 'same', 
                        kernel_initializer = kernel_initializer)
    
    b2 = conv2D_bn_relu(x, 
                        filters=(branch_filters // 3) * 2, 
                        kernel_size = 1, 
                        strides = 1,
                        padding = "same", 
                        kernel_initializer = kernel_initializer)
    b2 = conv2D_bn_relu(b2, 
                        filters = (branch_filters // 6) * 5, 
                        kernel_size = 3, 
                        strides = 1, 
                        padding = 'same', 
                        kernel_initializer = kernel_initializer)
    b2 = conv2D_bn_relu(b2, 
                        filters = branch_filters, 
                        kernel_size = 3, 
                        strides = 1, 
                        padding = 'same', 
                        kernel_initializer = kernel_initializer)
    
    pool = tf.keras.layers.MaxPooling2D(pool_size = (2, 2), strides = 1, padding = 'same')(x)
    a = tf.keras.layers.concatenate([b1, b2, pool])
    return a

def build_inception(x_train):

    # This function builds the Inception model.

    stem_width = 128

    inputs = tf.keras.layers.Input(shape = x_train.shape[1:])
    x = conv2D_bn_relu(inputs,
                    filters = stem_width,
                    kernel_size = 5,
                    strides = 1,
                    padding = 'same',
                    name = 'conv_1')

    x = reduction_module_A(x, filters = int(2*stem_width))
    x = tf.keras.layers.SpatialDropout2D(0.3)(x)

    x = inception_module_A(x, filters = int(2*stem_width))
    x = inception_module_A(x, filters = int(2*stem_width))

    x = reduction_module_A(x, filters = int(3*stem_width))
    x = tf.keras.layers.SpatialDropout2D(0.5)(x)

    x = inception_module_C(x, filters = int(3*stem_width))
    x = inception_module_C(x, filters = int(3*stem_width))
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Dense(10, name = 'logits')(x)
    x = tf.keras.layers.Activation('softmax', name = 'softmax')(x)

    tf.keras.backend.clear_session()
    model = tf.keras.models.Model(inputs = inputs, outputs = x)
    return model

if __name__ == "__main__":

    # The following code trains and then evaluates the Inception model on either
    # MNIST of Fashion MNIST as specified.

    fashion = False

    data_function = data.get_fashion_data if fashion else data.get_mnist_data
    ((x_train, y_train), (x_test, y_test)) = data_function(border = False, size = (8, 8))
    x_train = tf.reshape(x_train, [-1, 8, 8])
    x_test = tf.reshape(x_test, [-1, 8, 8])
    x_train = tf.tile(x_train[..., None], (1, 1, 1, 3))
    x_test = tf.tile(x_test[..., None], (1, 1, 1, 3))

    model = build_inception(x_train)
    callbacks = [tf.keras.callbacks.EarlyStopping("val_accuracy", patience = 20, restore_best_weights = True)]
    model.compile(loss = 'categorical_crossentropy', 
                    optimizer = tf.keras.optimizers.Adamax(learning_rate = 0.006, beta_1 = 0.49, beta_2 = 0.999),
                    metrics = ['accuracy'])
    model.fit(x_train, y_train, 64, 1, validation_split = 1/7, callbacks = callbacks, verbose = 1)
    score = model.evaluate(x_test, y_test, verbose = 0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
