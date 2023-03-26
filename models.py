import pathlib

import tensorflow as tf

# This module contains the custom layer and model classes that
# are used to construct the tensor network regression models.

class Pow_Feat_Layer(tf.keras.layers.Layer):

    # This class represents the layer which applies a multilinear 
    # tensor product featurization to the image pixels. The vector
    # components are given by [1, x, ..., x**max_power].

    def __init__(self, max_power, dtype = "float32"):
        super().__init__(dtype = dtype)
        self.powers = tf.range(max_power + 1, dtype = dtype)[None, None]

    def call(self, inputs):
        output = inputs[..., None] ** self.powers
        return output    

class Decomp_Feat_Layer(tf.keras.layers.Layer):

    # This class represents the layer which prepares the data for 
    # a subsequent interaction decomposition. The only difference between
    # this layer and the Pow_Feat layer is that the max_power is capped at
    # 1, and an extra dimension is added to the vector components.

    def __init__(self, dtype = "float32"):
        super().__init__(dtype = dtype)

    def call(self, inputs):
        diag = tf.stack([tf.ones_like(inputs), inputs], -1)
        matrices = tf.linalg.diag(diag)
        return matrices

class MPS_Layer(tf.keras.layers.Layer):

    # This layer represents parameterized layer of the MPS regression
    # model. It is initialized using a specified bond dimension and
    # number of class labels, and automatically initializes to the shape 
    # of the data. 

    def __init__(self, bond_dim, num_classes, dtype = "float32"):
        super().__init__(dtype = dtype)
        self.bond_dim = bond_dim
        self.num_classes = num_classes
        self.set_decomp(False)

    def build(self, input_shape):
        (_, num_sites, phy_dim) = input_shape[:3]
        self.num_sites = num_sites
        self.split = tf.Variable(num_sites // 2, trainable = False) # The output is placed in the middle
        self.matrix_weights = self.add_weight("matrix_weights",
            [phy_dim, num_sites, self.bond_dim, self.bond_dim], self.dtype, self.initializer)
        self.middle = self.add_weight("middle", # This tensor is the output component of the network
            [self.num_classes, self.bond_dim, self.bond_dim], self.dtype, self.middle_initializer)

    def contract(self, inputs, **kwargs):

        # This function generates a prediction on the passed input batch.

        split_data = tf.concat([inputs[:, self.split:], inputs[:, :self.split]], 1)
        matrices = tf.einsum("nij,jikl->inkl", split_data, self.matrix_weights)
        matrix_prod = self.reduction(matrices)
        outputs = tf.einsum("nkl,olk->no", matrix_prod, self.middle)
        return outputs

    def decomp(self, inputs, indices, **kwargs):

        # This function performs an interaction decomposition by 
        # contracting the network left-to-right and separating out
        # the different degree contributions.

        max_order = indices[-1].shape[0]
        split_data = tf.concat([inputs[:, self.split:], inputs[:, :self.split]], 1)
        order_matrices = tf.einsum("nsrj,jskl->rnskl", split_data, self.matrix_weights)
        cuml = order_matrices[:, :, 0]
        for i in range(1, self.num_sites):
            order_matrix = order_matrices[:, :, i]
            contract = tf.einsum("rnkl,qnlm->qrnkm", cuml, order_matrix)
            combined = contract[0, 1:] + contract[1, :-1]
            cuml = tf.concat([contract[0, :1], combined, contract[1, -1:]], 0)[:max_order]
        order_output = tf.einsum("rnlm,oml->nor", cuml, self.middle)
        return order_output

    def set_decomp(self, true_false):

        # This function determines whether the model should perform
        # an interaction decomposition (True) or a normal contraction
        # (False).

        if true_false:
            setattr(self, "call", self.decomp)
        else:
            setattr(self, "call", self.contract)

    @staticmethod
    def initializer(shape, dtype):

        # This function initializes the component tensors of the network.
        # The tensors need to be initialized such that they basically act 
        # like the identity.
        
        (phys_dim, num_sites, bond_dim, bond_dim) = shape
        bias = tf.tile(tf.eye(bond_dim, dtype = dtype)[None, None], (1, num_sites, 1, 1))
        kernel = tf.random.normal([phys_dim - 1, num_sites, bond_dim, bond_dim], 0, 1e-2, dtype)
        weights = tf.concat([bias, kernel], 0)
        return weights

    @staticmethod
    def middle_initializer(shape, dtype):
        
        # This function initializes the output component tensor.

        (num_sites, bond_dim, bond_dim) = shape
        weights = tf.tile([tf.eye(bond_dim, dtype = dtype)], (num_sites, 1, 1))
        noised = weights + tf.random.normal(weights.shape, 0, 1e-2, dtype = dtype)
        return noised

    @staticmethod
    def reduction(tensor):

        # This function performs an efficient contraction of the MPS
        # component matrices generated by contraction with the data
        # vectors.

        size = int(tensor.shape[0])
        while size > 1:
            half_size = size // 2
            nice_size = 2 * half_size
            leftover = tensor[nice_size:]
            tensor = tf.matmul(tensor[0:nice_size:2], tensor[1:nice_size:2])
            tensor = tf.concat([tensor, leftover], axis=0)
            size = half_size + int(size % 2 == 1)
        return tensor[0]
    
class TTN_Layer(tf.keras.layers.Layer):

    # This layer represents parameterized layer of the TTN regression
    # model. It is initialized using a specified bond dimension and
    # number of class labels, and automatically initializes to the shape 
    # of the data.  

    def __init__(self, bond_dim, num_classes, dtype = "float32"):
        super().__init__(dtype = dtype)
        self.bond_dim = bond_dim
        self.num_classes = num_classes
        self.set_decomp(False)

    def build(self, input_shape):

        # This function builds the network based on the number of 
        # features, with log2(features) different layers.

        (_, num_sites, phy_dim) = input_shape[:3]
        log2 = tf.math.log(tf.cast(num_sites, "float32")) / tf.math.log(2.0)
        num_layers = tf.cast(tf.math.ceil(log2), "int32")
        self.weight_list = self.get_weight_list(num_layers, phy_dim)
        self.norm = tf.Variable(self.get_weight_norm(), trainable = False)
    
    def contract(self, inputs, **kwargs):

        # This function generates a prediction on the passed input batch.

        output = inputs
        for weight in self.weight_list:
            even = output[:, ::2]
            odd = output[:, 1::2]
            output = tf.einsum("aijk,nai,naj->nak", weight, even, odd)
        return output[:, 0] / self.norm

    def decomp(self, inputs, indices, **kwargs):

        # This function performs an interaction decomposition by pairing
        # up tensors in each layer and contracting from the bottom up.

        output = tf.transpose(inputs, [2, 3, 0, 1])
        for (weight, index) in zip(self.weight_list, indices):
            pairs = tf.stack([output[...,::2], output[...,1::2]])
            order_slices = tf.gather_nd(pairs, index)
            left_slices = tf.expand_dims(order_slices[:, :, 0], 3)
            right_slices = tf.expand_dims(order_slices[:, :, 1], 2)
            prod_sum = tf.reduce_sum(left_slices * right_slices, 1)
            output = tf.einsum("rijns,sijk->rkns", prod_sum, weight)
        order_outputs = tf.transpose(output[:, :, :, 0], [2, 1, 0]) / self.norm
        return order_outputs

    def get_weight_list(self, num_layers, phy_dim):

        # This function generates the parameter tensors of the model based
        # on the number of layers.

        if num_layers > 1:
            first_layer = self.create_weight(2**num_layers, phy_dim, self.bond_dim, base = True)
            middle = [self.create_weight(2**i, self.bond_dim, self.bond_dim) for i in reversed(range(2, num_layers))]
            last_layer = self.create_weight(2, self.bond_dim, self.num_classes)
            weight_list = [first_layer] + middle + [last_layer]
        else:
            weight_list = [self.create_weight(2, phy_dim, self.num_classes, base = True)]
        return weight_list

    def create_weight(self, num_sites, input_dim, output_dim, base = False):
        init = self.initializer_base if base else self.initializer_stoch
        num_nodes = num_sites // 2
        shape = [num_nodes, input_dim, input_dim, output_dim]
        weight = self.add_weight(f"layer_{num_sites}", shape, self.dtype, init)
        return weight

    def get_weight_norm(self):

        # This function computes the norm of the network in order to 
        # generate a numerically-stable output.

        output = tf.reduce_sum(self.weight_list[0], [1, 2])
        for weight in self.weight_list[1:]:
            even = output[::2]
            odd = output[1::2]
            output = tf.einsum("aijk,ai,aj->ak", weight, even, odd)
        norm = tf.exp(tf.reduce_mean(tf.math.log(tf.abs(output))))
        return norm

    def set_decomp(self, true_false):

        # This function determines whether the model should perform
        # an interaction decomposition (True) or a normal contraction
        # (False).

        if true_false:
            setattr(self, "call", self.decomp)
        else:
            setattr(self, "call", self.contract)

    @staticmethod
    def initializer_base(shape, dtype):

        # This function initializes the bottom layer of the tree
        # by setting a single vector of elements in the tensor to 
        # one.

        (num_nodes, input_dim, _, output_dim) = shape
        template = tf.scatter_nd([[0, 0, 0]], tf.constant([1], dtype), [input_dim, input_dim, output_dim])
        weights = tf.tile(template[None], [num_nodes, 1, 1, 1])
        noised = weights + tf.random.normal(weights.shape, 0, 1e-1, dtype = dtype)
        return noised

    @staticmethod
    def initializer_stoch(shape, dtype):

        # This function initializes the tensors in layers above the
        # bottom layer as stochastic tensors, such that the L1 norm
        # of the input is maintained at 1.

        (num_nodes, input_dim, _, output_dim) = shape
        weights = tf.random.uniform([num_nodes, input_dim, input_dim, output_dim - 1], -1, 1, dtype = dtype)
        norms = tf.reduce_sum(weights, -1, keepdims = True)
        normalized = tf.concat([weights, 1 - norms], -1)
        return normalized

class TN_Model(tf.keras.Model):

    # This class is the Keras model which contains the tensor network
    # and featurization layers. It can be initialized with either the MPS
    # layer or the TTN layer, and will use either featurization layer depending
    # on whether it is performing an interaction decomposition.

    def __init__(self, network, bond_dim, num_classes, dtype):
        super().__init__()
        self.flatten_layer = tf.keras.layers.Flatten(dtype = dtype)
        self.pow_layer = Pow_Feat_Layer(1, dtype)
        self.decomp_feat_layer = Decomp_Feat_Layer(dtype)
        self.tn_layer = self.get_layer(network, bond_dim, num_classes, dtype)
        self.set_output(False, True)
        self.indices = None
        self.order_list = -1
        self.num_features = None

    def build(self, input_shape):
        self.num_features = input_shape[-1]
        self.set_order(self.order_list)
        super().build(input_shape)

    def combine(self, inputs, **kwargs):

        # This function generates a normal prediction on the
        # inputs.

        inputs = self.flatten_layer(inputs)
        pows = self.pow_layer(inputs)
        outputs = self.tn_layer(pows)
        return outputs

    def decomp(self, inputs, **kwargs):

        # This function perform an interaction decomposition
        # over the inputs.

        inputs = self.flatten_layer(inputs)
        feat = self.decomp_feat_layer(inputs)
        orders = self.tn_layer(feat, self.indices)
        outputs = tf.gather(orders, self.order_list, axis = -1)
        return outputs

    def decomp_summed(self, inputs, **kwargs):

        # This function perform an interaction decomposition and then
        # sums the contributions. This can differ from a normal contraction
        # if some of the degree contributions have been removed.

        inputs = self.flatten_layer(inputs)
        decomp = self.decomp(inputs)
        outputs = tf.reduce_sum(decomp, -1)
        return outputs

    def set_order(self, order):

        # This function sets the multilinear orders that will be 
        # preserved in the interaction decomposition.

        if self.num_features is not None:
            log2 = tf.math.log(tf.cast(self.num_features, "float32")) / tf.math.log(2.0)
            num_layers = tf.cast(tf.math.ceil(log2), "int32")
            if isinstance(order, int):
                order = self.num_features if order < 0 else order
                self.order_list = tf.range(order + 1)
            else:
                self.order_list = tf.convert_to_tensor(order)
            self.indices = get_ragged_indices(num_layers, tf.reduce_max(self.order_list))
        else:
            self.order_list = order

    def set_output(self, decomp, summed):
        self.tn_layer.set_decomp(decomp)
        if not decomp:
            setattr(self, "call", self.combine)
        else:
            if summed:
                setattr(self, "call", self.decomp_summed)
            else:
                setattr(self, "call", self.decomp)
                
    @staticmethod  
    def get_layer(name, bond_dim, num_classes, dtype):

        # This function determines what kind of network will be used
        # in the model.

        if name == "mps":
            layer = MPS_Layer(bond_dim, num_classes, dtype)
        elif name == "ttn":
            layer = TTN_Layer(bond_dim, num_classes, dtype)
        else:
            raise ValueError(f"Network '{name}' not recognized.")
        return layer

class Linear(tf.keras.Model):

    # This class is the Keras model representing a generic multilinear 
    # regression algorithm whose regression coefficients are not 
    # generated by a tensor network. The maximum order of the model is 
    # set upon initialization.

    def __init__(self, max_order, num_classes, dtype):
        super().__init__()
        self.set_output(False, True)
        self.bias = self.add_weight("bias", [1, num_classes], dtype, tf.keras.initializers.Zeros())
        self.dense_list = []
        for i in range(max_order):
            layer = tf.keras.layers.Dense(num_classes, dtype = dtype, use_bias = False, kernel_regularizer = tf.keras.regularizers.L1(0.000))
            self.dense_list.append(layer)
            setattr(self, f"dense_{i}", layer)
        self.flatten_layer = tf.keras.layers.Flatten(dtype = dtype)
        self.max_order = max_order
        self.order = max_order
        self.indices = None
        self.order_list = None

    def build(self, input_shape):
        num_features = input_shape[-1]
        self.indices = get_index_lists(num_features, self.order)
        self.set_order(self.order)
        super().build(input_shape)

    def combine(self, inputs, **kwargs):

        # This function generates a normal prediction on the
        # inputs.

        inputs = self.flatten_layer(inputs)
        outputs = tf.tile(self.bias, (tf.shape(inputs)[0], 1))
        for (i, dense) in enumerate(self.dense_list):
            feat_sets = tf.gather(tf.transpose(inputs), self.indices[i], axis = 0)
            products = tf.reduce_prod(feat_sets, 1)
            new_output = dense(tf.transpose(products))
            outputs = outputs + new_output
        return outputs

    def decomp(self, inputs, **kwargs):

        # This function separates the contributions from different
        # multilinear degrees. It is primarily included for compatability
        # with the TN_Model class.

        inputs = self.flatten_layer(inputs)
        outputs = [tf.tile(self.bias, (tf.shape(inputs)[0], 1))]
        for (i, dense) in enumerate(self.dense_list):
            feat_sets = tf.gather(tf.transpose(inputs), self.indices[i], axis = 0)
            products = tf.reduce_prod(feat_sets, 1)
            new_output = dense(tf.transpose(products))
            outputs.append(new_output)
        orders = tf.stack(outputs, -1)
        outputs = tf.gather(orders, self.order_list, axis = -1)
        return outputs

    def decomp_summed(self, inputs, **kwargs):

        # This function sums the different multinear contributions. It
        # is primarily included for compatability with the TN_Model class.

        inputs = self.flatten_layer(inputs)
        decomp = self.decomp(inputs)
        outputs = tf.reduce_sum(decomp, -1)
        return outputs

    def set_order(self, order):

        # This function sets the allowed orders for the multilinear
        # regression.

        if isinstance(order, int):
            order = self.max_order if order < 0 else order
            self.order_list = tf.range(order + 1)
        else:
            self.order_list = tf.convert_to_tensor(order)
        self.order = tf.reduce_max(self.order_list)
        if self.order > self.max_order:
            raise ValueError("Order cannot be greater than max_order.")

    def set_output(self, decomp, summed):
        if not decomp:
            setattr(self, "call", self.combine)
        else:
            if summed:
                setattr(self, "call", self.decomp_summed)
            else:
                setattr(self, "call", self.decomp)         

def get_ragged_indices(num_layers, order_limit = -1):

    # This function generates the indices that are used to generate
    # the ragged tensors in the TTN layer interaction decomposition.
    # They work by pairing up different slices of the intermediate
    # tensors generated during a contraction, such each pair sums
    # to the specified degree.  

    if order_limit < 0:
        order_limit = 2**num_layers
    ragged_tensors = []
    prev_max_order = 1
    for i in range(1, num_layers + 1):
        max_order = 2**i
        order_tensor = tf.range(prev_max_order + 1)
        pairs = tf.stack(tf.meshgrid(order_tensor, order_tensor), -1)
        flat_pairs = tf.reshape(pairs, [-1, 2])
        pair_sums = tf.reduce_sum(flat_pairs, 1)
        order_tensors = []
        sizes = []
        for order in range(min(max_order, order_limit) + 1):
            order_pairs = tf.boolean_mask(flat_pairs, pair_sums == order)
            num_pairs = tf.shape(order_pairs)[0]
            pair_index = tf.tile([[0, 1]], (num_pairs, 1))
            index_tensors = tf.stack([pair_index, order_pairs], -1)
            order_tensors += tf.unstack(index_tensors)
            sizes.append(num_pairs)
        ragged = tf.RaggedTensor.from_row_lengths(order_tensors, sizes)
        ragged_tensors.append(ragged)
        prev_max_order = max_order
    return ragged_tensors

def get_index_lists(num_features, order):

    # This function generates ragged index sets for the
    # specified order.

    index_lists = []
    for order in range(1, order + 1):
        new_list = []
        get_indices(num_features, order - 1, 0, num_features - order + 1, [], new_list)
        tensor = tf.RaggedTensor.from_uniform_row_length(tf.reshape(new_list, [-1]), len(new_list[0]))
        index_lists.append(tensor)
    sizes = [tensor.nrows() for tensor in index_lists]
    values = tf.concat(index_lists, 0)
    index_tensor = tf.RaggedTensor.from_row_lengths(values, sizes)   
    return index_tensor

def get_indices(num_features, pwr, start, stop, cuml_list, index_list):

    # This function recursively gathers all index lists required for 
    # a given number of features. 

    for i in range(start, stop):
        new_cuml_list = cuml_list.copy()
        new_cuml_list.append(i)
        if stop < num_features:
            get_indices(num_features, pwr - 1, i + 1, stop + 1, new_cuml_list, index_list)
        else:
            index_list.append(new_cuml_list)

def get_model(name, num_classes, dtype, bond_dim = None, order = None):

    # This function retreives a model of the specified type.

    if name == "mps":
        model = TN_Model("mps", bond_dim, num_classes, dtype)
    elif name == "ttn":
        model = TN_Model("ttn", bond_dim, num_classes, dtype)
    elif name == "linear":
        model = Linear(order, num_classes, dtype)
    else:
        raise ValueError(f"Model '{name}' not recognized.")
    return model

def load_model(weight_dir):

    # This function instantiates a model based on a set of saved weights.

    index_path = next(pathlib.Path(weight_dir).glob("*.index")).stem
    (name, class_string, order_bond, dtype) = index_path.split("_")
    model = get_model(name, int(class_string), dtype, int(order_bond), int(order_bond))
    model.load_weights(weight_dir + "/" + index_path)
    return model
