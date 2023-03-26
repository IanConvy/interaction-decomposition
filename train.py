import configparser
import ast

import tensorflow as tf

import data
import models
import factor

# This module runs numerical experiments which optimize and evaluate the
# tensor network regression models.

def train_model(config, path_modifier = ""):

    # This functions trains and evaluates a model based on the specified 
    # settings.

    size = (config["size"], config["size"])
    ((train_x, train_y), (test_x, test_y), num_classes) = data.get_dataset(config["dataset"], size = size, border = False, shuffle = False)
    (name, dtype, bond_dim, max_order) = (config["model"], config["dtype"], config["bond_dim"], config["max_order"])
    model = models.get_model(name, num_classes, dtype, bond_dim, max_order)
    model.set_output(config["decomp"], True)
    if config["decomp"]:
        model.set_order(max_order)

    callbacks = [tf.keras.callbacks.EarlyStopping("val_accuracy", patience = config["wait"], restore_best_weights = True)]
    model.compile(loss = "mse",
        optimizer = get_optm(config["optm"], config["rate"]),
        metrics = ['accuracy'],
        run_eagerly = False)
    model.fit(train_x, train_y,
        batch_size = config["batch_size"],
        epochs = config["epochs"],
        verbose = 1,
        validation_split = config["val_split"],
        callbacks = callbacks)

    score = model.evaluate(test_x, test_y, verbose = 0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    if config["save_path"]:
        save_path = "saved/" + config["save_path"]
        if path_modifier:
            save_path = save_path + "__" + path_modifier
        if name == "linear":
            model_string = f"/linear_{num_classes}_{max_order}_{dtype}"
        else:
            model_string = f"/{name}_{num_classes}_{bond_dim}_{dtype}"
        model.save_weights(save_path + model_string, save_format = "tf")
        if config["factor"]:
            factor.factorize(save_path, (test_x, test_y))

def get_optm(optm_type, rate):

    # This function returns the specified optimizer.

    if optm_type == "rmsprop":
        optm = tf.keras.optimizers.RMSprop(rate)
    elif optm_type == "adam":
        optm = tf.keras.optimizers.Adam(rate)
    else:
        raise ValueError(f"Optimizer type '{optm_type}' not recognized.")
    return optm

def get_config_list(config):

    # This functions parses the train.ini file that is used to
    # specify the optimization and evaluation settings.

    config = {key:ast.literal_eval(val) for (key, val) in config.items()}
    reps = config.pop("reps")
    list_dict = {}
    num_trials = max([len(val) for val in config.values()])
    for (key, val_list) in config.items():
        len_diff = num_trials - len(val_list)
        val_list += len_diff*val_list[-1:]
        list_dict[key] = val_list
    config_list = [{key:val[i] for (key, val) in list_dict.items()} for i in range(num_trials)]
    return (config_list, reps)

# The following code loads the train.ini file and initiates the
# specified procedure.

parser = configparser.ConfigParser()
parser.read("train.ini")
(config_list, reps) = get_config_list(parser["train"])
if reps > 1:
    for i in range(reps):
        print(f"Rep: {i}")
        for config in config_list:
            train_model(config, str(i))
else:
    for config in config_list:
        train_model(config)
