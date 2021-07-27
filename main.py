_author_ = "davidbonet"
"""CW-DeepNNK, DeepNNK and Validation-based early stopping"""

import os, json
import numpy as np
from absl import flags, app
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
from utils.data import get_data
from utils.graph_utils import nnk_loo
import utils.tensorflow_utils as tf_utils
from utils.BatchDatasetReader import BatchDataset
from sklearn.model_selection import train_test_split
from tensorflow.python.client import device_lib

devices = device_lib.list_local_devices()

FLAGS = flags.FLAGS
flags.DEFINE_string("mode", "train", "train/test mode")
flags.DEFINE_string("experiment_folder", "", "Subfolder name for the experiment")
flags.DEFINE_integer("seed", 0, "Random seed for reproducibility")
# Data
flags.DEFINE_string("dataset", "cifar10", "Choose dataset (mnist/fasion_mnist/cifar10)")
flags.DEFINE_integer("num_classes", 2, "Num classes from chosen dataset (2 or 10)")
flags.DEFINE_integer("labeled_samples", 10000,
                     "Number of labeled samples to use in train and validation set")
flags.DEFINE_float("validation_percent", 0, 
                   "Fraction of labelled data to use for validation")
# Hyperparameters
flags.DEFINE_integer("epochs", 400, "Max number of epochs")
flags.DEFINE_float("lr", 0.001, "Learning rate")
flags.DEFINE_integer("batch_size", 50, "Batch size")
flags.DEFINE_bool("regularize", False, "Use dropout")
flags.DEFINE_string("optimizer", "Adam", "Optimizer (Adam/Momentum/GD)")
flags.DEFINE_string("weight_initializer","he_uniform",
                    "Weight initialization method (glorot_uniform/glorot_normal/he_normal/he_uniform)")
# Early stopping
flags.DEFINE_string("stopping", "None", 
                    "Early stopping method (cwdeepnnk/deepnnk/validation)")
flags.DEFINE_integer("criterion_freq", 1, "Compute stopping criterion every X epochs")
flags.DEFINE_integer("patience", 20, 
                     "Number of times to observe worsening generalization estimate before stopping")
# Graph related parameters
flags.DEFINE_integer("knn_param", 25, "Number of initial neighbors for NNK")
flags.DEFINE_string("kernel", "cosine", 
                    "Kernel for NNK graph construction (cosine/gaussian)")
flags.DEFINE_float(
    "interpol_queries", 1.0, 
    "Fraction of training set samples to use as queries in the LOO procedure")


def main(arg=None):
    tf.config.optimizer.set_jit(True)  # Enable XLA.
    tf.logging.set_verbosity("ERROR")
    np.seterr(divide="ignore", invalid="ignore")

    # Setting seed for reproducibility
    seed_value = FLAGS.seed
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    np.random.seed(seed_value)
    tf.random.set_random_seed(seed_value)
    tf.set_random_seed(seed_value)

    # Device
    for device in devices:
        if len(devices) > 1 and "CPU" in device.name:
            continue
        print("Using device: ", device.name)

    # Data
    dataset = FLAGS.dataset
    num_classes = FLAGS.num_classes
    x_train, x_test, y_train, y_test = get_data(dataset, num_classes)
    x_train, y_train = tf_utils.permute_data(x_train, y_train)
    image_shape = [x_train.shape[1], x_train.shape[2], x_train.shape[3]]

    # Random subsampling of labeled data
    labeled_samples = FLAGS.labeled_samples
    if labeled_samples > x_train.shape[0]:
        raise Exception(
            "Selected number of labeled samples is larger than training set"
        )
    elif labeled_samples < x_train.shape[0]:
        x_train, _, y_train, _ = train_test_split(
            x_train,
            y_train,
            train_size=float(labeled_samples / x_train.shape[0]),
            random_state=seed_value,
            shuffle=True,
            stratify=y_train,
        )

    # Split train and validation set
    val_percent = FLAGS.validation_percent
    if val_percent > 0:
        x_train, x_val, y_train, y_val = train_test_split(
            x_train,
            y_train,
            test_size=val_percent,
            random_state=seed_value,
            shuffle=True,
            stratify=y_train,
        )
        validation_dataset = BatchDataset(images=x_val, labels=y_val, labels_flag=True)
    train_dataset = BatchDataset(images=x_train, labels=y_train, labels_flag=True)
    test_dataset = BatchDataset(images=x_test, labels=y_test, labels_flag=True)

    criterion = FLAGS.stopping
    if "nnk" in criterion and val_percent > 0:
        raise Exception(
            "You are using a validation set, but you don't need it for NNK label interpolation"
        )
    elif criterion == "validation" and val_percent == 0:
        raise Exception(
            "You need to select a validation set for Validation-based early stopping"
        )

    print(f"Train shape: {x_train.shape}")
    if val_percent > 0:
        print(f"Validation shape: {x_val.shape}")
    print(f"Test shape: {x_test.shape}")
    print(f"Labels: {np.unique(np.concatenate((y_train, y_test)))}\n\n")

    # Model
    activation_dict = {}
    num_channels = 5
    regularize = FLAGS.regularize
    input_data = tf.placeholder(
        tf.float32, shape=[None] + image_shape, name="input_images"
    )
    labels = tf.placeholder(tf.float32, shape=[None, num_classes], name="input_labels")
    dropout_rate = tf.placeholder_with_default(0.0, shape=[], name="dropout_rate")
    weight_initializer = FLAGS.weight_initializer
    with tf.variable_scope("network", reuse=False):
        # Group 0
        W = tf_utils.weight_variable(
            [5, 5, image_shape[2], num_channels], weight_initializer, name="W_conv0"
        )
        b = tf_utils.bias_variable([num_channels], weight_initializer, name="b_conv0")
        activation_dict[0] = tf_utils.dropout_layer(
            (tf.nn.relu(tf_utils.conv2d_basic_valid(input_data, W, b))),
            rate=dropout_rate,
            regularize=regularize,
        )
        # Group 1
        W = tf_utils.weight_variable(
            [5, 5, num_channels, num_channels], weight_initializer, name="W_conv1"
        )
        b = tf_utils.bias_variable([num_channels], weight_initializer, name="b_conv1")
        activation_dict[1] = tf_utils.dropout_layer(
            tf.nn.relu(tf_utils.conv2d_basic_valid(activation_dict[0], W, b)),
            rate=dropout_rate,
            regularize=regularize,
        )
        activation_dict[1] = tf_utils.max_pool_2x2(activation_dict[1])
        # Group 2
        W = tf_utils.weight_variable(
            [5, 5, num_channels, num_channels], weight_initializer, name="W_conv2"
        )
        b = tf_utils.bias_variable([num_channels], weight_initializer, name="b_conv2")
        activation_dict[2] = tf_utils.dropout_layer(
            tf.nn.relu(tf_utils.conv2d_basic_valid(activation_dict[1], W, b)),
            rate=dropout_rate,
            regularize=regularize,
        )
        # Group 3
        W_0 = tf_utils.weight_variable(
            [3, 3, num_channels, 1], weight_initializer, name="W_conv3_ch0"
        )
        W_1 = tf_utils.weight_variable(
            [3, 3, num_channels, 1], weight_initializer, name="W_conv3_ch1"
        )
        W_2 = tf_utils.weight_variable(
            [3, 3, num_channels, 1], weight_initializer, name="W_conv3_ch2"
        )
        W_3 = tf_utils.weight_variable(
            [3, 3, num_channels, 1], weight_initializer, name="W_conv3_ch3"
        )
        W_4 = tf_utils.weight_variable(
            [3, 3, num_channels, 1], weight_initializer, name="W_conv3_ch4"
        )
        W_all = tf.concat([W_0, W_1, W_2, W_3, W_4], axis=3)
        last_layer_weights = [W_0, W_1, W_2, W_3, W_4]
        b_0 = tf_utils.bias_variable([1], weight_initializer, name="b_conv3_ch0")
        b_1 = tf_utils.bias_variable([1], weight_initializer, name="b_conv3_ch1")
        b_2 = tf_utils.bias_variable([1], weight_initializer, name="b_conv3_ch2")
        b_3 = tf_utils.bias_variable([1], weight_initializer, name="b_conv3_ch3")
        b_4 = tf_utils.bias_variable([1], weight_initializer, name="b_conv3_ch4")
        b_all = tf.concat([b_0, b_1, b_2, b_3, b_4], axis=0)
        last_layer_biases = [b_0, b_1, b_2, b_3, b_4]
        activation_dict[3] = tf_utils.dropout_layer(
            tf.nn.relu(tf_utils.conv2d_basic_valid(activation_dict[2], W_all, b_all)),
            rate=dropout_rate,
            regularize=regularize,
        )
        activation_dict[3] = tf_utils.max_pool_2x2(activation_dict[3])
    print("\nNetwork:")
    for i in range(len(activation_dict)):
        print(activation_dict[i])
    net = activation_dict
    if dataset == "cifar10":
        net_size = 9 * num_channels
    else:
        net_size = 4 * num_channels
    net_flatten = tf.reshape(net[3], [-1, net_size])
    W_fc1 = tf_utils.weight_variable(
        [net_size, num_classes], weight_initializer, name="W_fc1"
    )
    b_fc1 = tf_utils.bias_variable([num_classes], weight_initializer, name="b_fc1")
    logits = tf.matmul(net_flatten, W_fc1) + b_fc1
    pred = tf.nn.softmax(logits)
    x_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    )
    tf.summary.scalar("X-entropy", x_entropy)
    loss = x_entropy
    accuracy_op = tf_utils.model_accuracy(pred=pred, labels=labels)
    optim = tf_utils._get_optimizer(FLAGS.lr, optim=FLAGS.optimizer)
    train_variables = tf.trainable_variables()
    train_op = tf_utils._train(loss, train_variables, optim)

    epochs = FLAGS.epochs
    batch_size = FLAGS.batch_size
    sess = tf.Session(graph=tf.get_default_graph())
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=epochs)

    # Output directory
    name_experiment = f"ConvNet_{dataset}_num_classes_{num_classes}_optim_{FLAGS.optimizer}_lr_{FLAGS.lr}_batch_size"\
        f"_{batch_size}_labeled_samples_{labeled_samples}_val_percent_{val_percent}_stop_{criterion}_seed_{seed_value}"
    if criterion != "None":
        name_experiment += (
            f"_patience_{FLAGS.patience}_criterion_freq_{FLAGS.criterion_freq}"
        )
    if "nnk" in criterion:
        name_experiment += f"_kernel_{FLAGS.kernel}_knn_{FLAGS.knn_param}_interpol_queries_{FLAGS.interpol_queries}"
    experiment_folder = os.path.join("logs", FLAGS.experiment_folder)
    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder)
    model_output_folder = os.path.join(experiment_folder, name_experiment)
    if not os.path.exists(model_output_folder):
        os.makedirs(model_output_folder)
    ckpts_folder = os.path.join(model_output_folder, "ckpts")
    ckpt = tf.train.get_checkpoint_state(ckpts_folder)
    mode = FLAGS.mode

    def get_performance(dataset):
        dataset_size = dataset.get_dataset_size()
        n_batches = dataset_size // batch_size
        last_batch = dataset_size % batch_size
        if n_batches == 0:
            return 0, 0
        loss_val = np.zeros(n_batches, dtype=np.float)
        accuracy = np.zeros(n_batches, dtype=np.float)
        start_idx = 0
        for itr in range(n_batches):
            end_idx = start_idx + batch_size
            feed_dict = {
                input_data: dataset.images[start_idx:end_idx],
                labels: dataset.labels[start_idx:end_idx],
            }
            loss_val[itr], acc = sess.run([loss, accuracy_op], feed_dict=feed_dict)
            accuracy[itr] = np.mean(acc)
            start_idx = end_idx
        return np.mean(loss_val), np.mean(accuracy)

    # Train
    if mode == "train":
        if os.path.exists(os.path.join(ckpts_folder, "checkpoint")):
            raise Exception(
                f"ckpt directory is not empty. Remove previous training history.\nckpt directory: {ckpts_folder}"
            )
        file_csv = os.path.join(model_output_folder, "training.csv")
        with open(file_csv, "w+") as f_csv:
            print(
                "Epoch,Train_Loss,Train_Acc,Val_Loss,Val_Acc,Stopping_Metric,Patience,Best_Val,Channel",
                file=f_csv,
            )

        rate = 0.2 if regularize else 0
        train_size = len(train_dataset.images)
        interpol_queries = int(train_size * FLAGS.interpol_queries)

        # Stopping criterion parameters
        knn_param = FLAGS.knn_param
        kernel = FLAGS.kernel
        if criterion == "cwdeepnnk":
            ch_patience = np.ones(num_channels, dtype=np.int16) * FLAGS.patience
            best_val = np.ones(num_channels) * np.inf
        else:
            patience = FLAGS.patience
            best_val = np.inf

        for epoch in range(epochs):
            print("\nTraining...")
            train_dataset.reset_batch_offset()
            train_dataset.permute_data()
            n_batches = int(train_dataset.n_samples / batch_size)
            train_loss = np.zeros(n_batches, dtype=np.float)
            train_acc = np.zeros(n_batches, dtype=np.float)
            for itr in range(n_batches):
                batch_images, batch_labels = train_dataset.next_batch(
                    batch_size=batch_size
                )
                feed_dict = {
                    input_data: batch_images,
                    labels: batch_labels,
                    dropout_rate: rate,
                }
                _, train_loss[itr], acc = sess.run(
                    [train_op, loss, accuracy_op], feed_dict=feed_dict
                )
                train_acc[itr] = np.mean(acc)
                if itr % 50 == 0:
                    print(
                        f"Epoch: {epoch+1},\tItr: {itr}/{n_batches},  \tLoss: {train_loss[itr]:.4f},\tAcc: {train_acc[itr]:.4f}"
                    )

            if val_percent > 0:
                val_loss, val_acc = get_performance(validation_dataset)
                val_error = 1 - np.mean(val_acc)

            # Stopping criterion
            if epoch % FLAGS.criterion_freq == 0:
                if criterion == "cwdeepnnk":
                    for channel in range(0, num_channels):
                        if ch_patience[channel] > 0:
                            print(f"\nChannel {channel}:")
                            vector = net[3][:, :, :, channel]
                            d = tf_utils.get_tensor_size(vector)
                            activations_train = sess.run(
                                [vector],
                                feed_dict={
                                    input_data: train_dataset.images,
                                    labels: train_dataset.labels,
                                },
                            )
                            activations_train = np.reshape(
                                activations_train, [train_size, d]
                            )
                            # NNK LOO label interpolation error
                            ch_error = nnk_loo(
                                activations=activations_train,
                                labels=train_dataset.labels,
                                interpol_queries=interpol_queries,
                                knn_param=knn_param,
                                kernel=kernel,
                            )
                            if best_val[channel] <= ch_error:
                                ch_patience[channel] -= 1
                                print(
                                    f"NNK LOO error did not improve: {ch_error:.3f} vs. {best_val[channel]:.3f}. Patience {ch_patience[channel]}/{FLAGS.patience}"
                                )
                            else:
                                # Reset channel patience
                                print(
                                    f"NNK LOO error improved {best_val[channel]:.3f} -> {ch_error:.3f}"
                                )
                                best_val[channel] = ch_error
                                ch_patience[channel] = FLAGS.patience
                                name_ckpt = f"bestval_{epoch+1}_ch_{channel}"
                                saver.save(
                                    sess, ckpts_folder + f"/{name_ckpt}.ckpt", epoch + 1
                                )
                            # Stop training channel
                            if ch_patience[channel] == 0:
                                train_variables.remove(last_layer_weights[channel])
                                train_variables.remove(last_layer_biases[channel])
                                train_op = tf_utils._train(loss, train_variables, optim)
                            with open(file_csv, "a") as f_csv:
                                print(
                                    f"{int(epoch+1)},{np.mean(train_loss)},{np.mean(train_acc)},None,None,{ch_error},{ch_patience[channel]},{best_val[channel]},{channel}",
                                    file=f_csv,
                                )

                    if np.all(ch_patience == 0):
                        print("Breaking train loop: out of patience in all channels\n")
                        break

                elif criterion == "deepnnk":
                    vector = net[3][:, :, :, :]
                    d = tf_utils.get_tensor_size(vector)
                    activations_train = sess.run(
                        [vector],
                        feed_dict={
                            input_data: train_dataset.images,
                            labels: train_dataset.labels,
                        },
                    )
                    activations_train = np.reshape(activations_train, [train_size, d])
                    # NNK LOO label interpolation error
                    deepnnk_error = nnk_loo(
                        activations=activations_train,
                        labels=train_dataset.labels,
                        interpol_queries=interpol_queries,
                        knn_param=knn_param,
                        kernel=kernel,
                    )
                    if best_val <= deepnnk_error:
                        patience -= 1
                        print(
                            f"NNK LOO error did not improve: {deepnnk_error:.3f} vs. {best_val:.3f}. Patience {patience}/{FLAGS.patience}"
                        )
                        if patience <= 0:
                            print("Breaking train loop: out of patience\n")
                            break
                    else:
                        # Reset patience
                        print(
                            f"\nNNK LOO error improved {best_val:.3f} -> {deepnnk_error:.3f}"
                        )
                        best_val = deepnnk_error
                        patience = FLAGS.patience
                        name_ckpt = f"bestval_{epoch+1}"
                        saver.save(sess, ckpts_folder + f"/{name_ckpt}.ckpt", epoch + 1)
                    with open(file_csv, "a") as f_csv:
                        print(
                            f"{int(epoch+1)},{np.mean(train_loss)},{np.mean(train_acc)},None,None,{deepnnk_error},{patience},{best_val},None",
                            file=f_csv,
                        )

                elif criterion == "validation":
                    if best_val <= val_error:
                        patience -= 1
                        print(
                            f"Validation error did not improve: {val_error:.3f} vs. {best_val:.3f}. Patience {patience}/{FLAGS.patience}"
                        )
                        if patience <= 0:
                            print("Breaking train loop: out of patience\n")
                            break
                    else:
                        # Reset patience
                        print(
                            f"\Validation error improved {best_val:.3f} -> {val_error:.3f}"
                        )
                        best_val = val_error
                        patience = FLAGS.patience
                        name_ckpt = f"bestval_{epoch+1}"
                        saver.save(sess, ckpts_folder + f"/{name_ckpt}.ckpt", epoch + 1)
                    with open(file_csv, "a") as f_csv:
                        print(
                            f"{int(epoch+1)},{np.mean(train_loss)},{np.mean(train_acc)},{val_loss},{val_acc},{val_error},{patience},{best_val},None",
                            file=f_csv,
                        )

                # No stopping
                elif val_percent > 0:
                    with open(file_csv, "a") as f_csv:
                        print(
                            f"{int(epoch+1)},{np.mean(train_loss)},{np.mean(train_acc)},{val_loss},{val_acc},None,None,None,None",
                            file=f_csv,
                        )
                else:
                    with open(file_csv, "a") as f_csv:
                        print(
                            f"{int(epoch+1)},{np.mean(train_loss)},{np.mean(train_acc)},None,None,None,None,None,None",
                            file=f_csv,
                        )

    elif mode == "test":
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model restored ... %s" % ckpt.model_checkpoint_path)

        test_loss, test_accuracy = get_performance(test_dataset)
        print(
            "Test Data results - X-Entropy: %g, Accuracy %g"
            % (test_loss, test_accuracy)
        )
        train_loss, train_accuracy = get_performance(train_dataset)
        print(
            "Train Data results - X-Entropy: %g, Accuracy %g"
            % (train_loss, train_accuracy)
        )
        if val_percent > 0:
            validation_loss, validation_accuracy = get_performance(validation_dataset)
            print(
                "Validation Data results - X-Entropy: %g, Accuracy %g"
                % (validation_loss, validation_accuracy)
            )
        else:
            validation_loss, validation_accuracy = None, None

        output_results_file = os.path.join(model_output_folder, "results.json")
        with open(output_results_file, "w") as f:
            results = {
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
            }
            json.dump(results, f)
    else:
        raise Exception("Unknown mode: " + mode)


if __name__ == "__main__":
    app.run(main)
