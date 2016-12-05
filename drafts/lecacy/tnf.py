"""
#TODO:
    Nesterov
    different batch sizes
    filter_size > len(x)
"""

import tensorflow as tf

import time
import numpy as np
from sklearn.cross_validation import KFold, StratifiedKFold
from tabulate import tabulate
from collections import OrderedDict, defaultdict
from evolutron.networks.tnf import tfDREAM


class DeepTrainer:
    def __init__(self,
                 #net,
                 loss_objective_fn=None,
                 classification=False,
                 max_epochs=2000,
                 verbose=False,
                 batch_size=8,
                 learning_rate=0.01,
                 patience=True, filters=10, filter_size=20,
                 num_conv_layers=1, num_fc_layers=2, keep_prob=0.8):
        #self.net = net
        #self.network = net.network
        self.loss_objective_fn = loss_objective_fn
        self.classification = classification
        self.max_epochs = max_epochs
        self.verbose = verbose
        self.learning_rate = learning_rate
        self.patience = patience
        self.filters = filters
        self.filter_size = filter_size
        self.num_conv_layers = num_conv_layers
        self.num_fc_layers = num_fc_layers
        self.keep_prob = keep_prob

        self.train_err_mem = []
        self.val_err_mem = []
        self.val_acc_mem = []
        self.train_acc_mem = []
        self.fold_val_losses = None
        self.fold_train_losses = None
        self.k_fold_history = defaultdict(list)
        self._funcs_init = False

        local_device_protos = tf.python.client.device_lib.list_local_devices()
        self.num_gpus = len([x.name for x in local_device_protos if x.device_type == 'GPU'])
        self.batch_size = max(self.num_gpus, batch_size)

    def _create_functions(self):
        if getattr(self, '_funcs_init', True):
            return

        # TODO: read inputs form net arch
        """
        input_layers = [layer for layer in lasagne.layers.get_all_layers(self.network)
                        if isinstance(layer, lasagne.layers.InputLayer)]

        inputs = [input_layer.input_var for input_layer in input_layers][0]

        """
        # loss, _, _ = self.net.build_loss()
        """
        val_loss, val_acc, val_prediction = self.net.build_loss()


        self.f = theano.function([inputs, self.net.targets],
                                 [prediction, self.net.targets])
        #  Debugging function

        params = lasagne.layers.get_all_params(self.network, trainable=True)

        updates = lasagne.updates.nesterov_momentum(loss, params,
                                                    learning_rate=self.learning_rate,
                                                    momentum=0.975)


        # Compile a function performing a training step on a mini-batch
        # ADAM insted of Nesterov!!!
        self.train_fn = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.975).minimize(
            self.net.build_loss()[0])


        # Compile a second function computing the validation loss and accuracy: (difference: Deterministic=True)
        self.val_fn = theano.function(inputs=[inputs, self.net.targets],
                                      outputs=['loss:0', 'acc:0'],
                                      allow_input_downcast=True)

        self.pred_fn = theano.function(inputs=[inputs],
                                       outputs=[val_prediction],
                                       allow_input_downcast=True)

        """
        tf.initialize_all_variables().run()

        self._funcs_init = True

    def tower_init(self, seq, DNAse, labels, masks, gpu_num, block_size):
        block_size = int(block_size/self.num_gpus)*self.num_gpus

        with tf.device('/gpu:%d' % gpu_num):
            # Uploading the data set to gpu memory
            gpu_seq = [[]]
            gpu_DNAse = [[]]
            gpu_labels = [[]]
            gpu_masks = [[]]
            for seq_batch, DNAse_batch, labels_batch, masks_batch in self._iterate_huge(seq, DNAse, labels, masks, block_size, gpu_num):
                try:
                    gpu_seq += seq_batch
                    gpu_DNAse += DNAse_batch
                    gpu_labels += labels_batch
                    gpu_masks += masks_batch
                except:
                    gpu_seq = seq_batch
                    gpu_DNAse = DNAse_batch
                    gpu_labels = labels_batch
                    gpu_masks = masks_batch

            # Building the model on the gpu
            gpu_network = tfDREAM(gpu_seq, gpu_DNAse, gpu_labels, gpu_masks, block_size/self.num_gpus,
                                  self.filters, self.filter_size, self.num_conv_layers,
                                  self.num_fc_layers, self.keep_prob)

            return gpu_network.build_loss()

    @staticmethod
    def _split_dataset(data_size, holdout, validate):

        test_idx = range(0, int(holdout * data_size))

        valid_idx = range(int(holdout * data_size), int((validate + holdout) * data_size))

        train_idx = range(int((validate + holdout) * data_size), data_size)

        return train_idx, valid_idx, test_idx

    @staticmethod
    def _iterate_minibatches(X, y, batch_size):
        assert len(X) == len(y)

        for start_idx in range(0, len(X), batch_size):
            excerpt = slice(start_idx, min(start_idx + batch_size, len(X)))
            yield X[excerpt], y[excerpt]  # TODO: yield excerpt

    def _iterate_huge(self, x, y, z, v, block_size, i):
        # TODO: here will implement shared variable storage and pass indexes
        assert len(x) == len(y)

        mini_batch_size = int(block_size/self.num_gpus)

        for start_idx in range(i*mini_batch_size, len(x), block_size):
            excerpt = slice(start_idx, min(start_idx + mini_batch_size, len(x)))
            yield x[excerpt], y[excerpt], z[excerpt], v[excerpt]

    def average_gradients(tower_grads):
        #Calculate the average gradient for each shared variable across all towers.

        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(0, grads)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)

        return average_grads

    def fit_dream(self, train_set_x, train_set_y, valid_set_x=None, valid_set_y=None, epochs=1):
        with tf.Graph().as_default(), tf.device('/cpu:0'):
            seq, DNAse = np.split(np.concatenate((train_set_x, valid_set_x)),[200],axis=1)
            labels, masks = np.split(np.concatenate((train_set_y, valid_set_y)),2,axis=1)
            train_size = train_set_x.shape[0]
            # Early-stopping parameters
            patience = 100000000  # look as this many examples regardless
            patience_increase = 2  # wait this times much longer when a new best is found
            improvement_threshold = 0.998  # a relative improvement of this much is considered significant
            best_validation_loss = np.inf
            it = 0
            done_looping = False

            # initializing towers
            tower_grads = []
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)

            for i in range(self.num_gpus):
                with tf.name_scope('tower_%d' %i) as scope:
                    _, loss, accuracy = self.tower_init(seq, DNAse, labels, masks, i, self.batch_size)

                    # Reuse variables for the next tower.
                    tf.get_variable_scope().reuse_variables()

                    # Calculate the gradients for the batch of data on this tower.
                    grads = opt.compute_gradients(loss)

                    # Keep track of the gradients across all towers.
                    tower_grads.append(grads)

            # synchronization point across all towers.
            grads = self.average_gradients(tower_grads)

            # Apply the gradients to adjust the shared variables.
            apply_gradient_op = opt.apply_gradients(grads)

            init = tf.initialize_all_variables()

            sess = tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=True))
            sess.run(init)

            epochs = min(epochs, self.max_epochs)
            epoch = 0
            start_time = time.time()
            print(tabulate([['Epoch', 'Train Loss', 'Train Acc', 'Val Loss', 'Val Acc', 'Time']], stralign='center',
                           headers="firstrow"))
            # We iterate over epochs:
            try:
                while epoch < epochs and not done_looping:
                    epoch += 1

                    # In each epoch, we do a full pass over the training data:
                    train_err = []
                    train_acc = []
                    epoch_time = time.time()
                    # count=0
                    for X, y in self._iterate_huge(train_set_x, train_set_y, self.batch_size):
                        err, acc = self.train_fn(X[:, :800].reshape((-1,4, 200)),
                                                 X[:, 800:].reshape((-1,1,100)),
                                                 y[:, :32],
                                                 y[:, 32:])
                        train_err.append(err)
                        train_acc.append(acc)
                        # pred, targ = self.f(X, y)
                        # print(pred)
                        # print(targ)
                        # print(pred.shape)
                        # print(targ.shape)
                        # count += 1
                        # print(count)

                    # After that, we do a full pass over the validation data:
                    val_err = []
                    val_acc = []
                    for X, y in self._iterate_huge(valid_set_x, valid_set_y, self.batch_size):
                        err, acc = self.val_fn(X[:, :800].reshape((-1,4, 200)),
                                                 X[:, 800:].reshape((-1,1,100)),
                                                 y[:, :32],
                                                 y[:, 32:])
                        val_err.append(err)
                        val_acc.append(acc)

                    self.train_err_mem.append(np.mean(train_err))
                    self.train_acc_mem.append(100 * np.mean(train_acc))
                    if len(valid_set_x) >0:
                        self.val_err_mem.append(np.mean(val_err))
                        self.val_acc_mem.append(100 * np.mean(val_acc))
                    else:
                        self.val_err_mem.append('-')
                        self.val_acc_mem.append('-')

                    print(tabulate([['Epoch', 'Train Loss', 'Train Acc', 'Val Loss', 'Val Acc', 'Time'],
                                    ["{}/{}".format(epoch, epochs),
                                     self.train_err_mem[-1], self.train_acc_mem[-1],
                                     self.val_err_mem[-1], self.val_acc_mem[-1],
                                     "{:.3f}s".format(time.time() - epoch_time)]],
                                   tablefmt='plain', floatfmt=".6f", stralign='center',
                                   headers="firstrow").rsplit('\n', 1)[-1])

                    # If we have the best validation score until now
                    if self.val_err_mem[-1] < best_validation_loss:

                        # Increase patience if loss improvement is good enough
                        if self.val_err_mem[-1] < best_validation_loss * improvement_threshold:
                            patience = max(patience, it * patience_increase)

                        # save best validation score and iteration number
                        best_validation_loss = self.val_err_mem[-1]

                    # print(self.get_all_param_values())

                    # Stop if above early stopping limit
                    it = (epoch - 1) * train_size
                    if patience <= it and self.patience:
                        done_looping = True
            except KeyboardInterrupt:
                return

        print('Model trained for {0} epochs. Total time: {1:.3f}s'.format(epoch, time.time() - start_time))

    def fit(self, x_data, y_data, epochs=1, holdout=.0, validate=.0, handle=''):
        sess = tf.get_default_session()

        assert (validate >= 0 and holdout >= 0)

        # Merge all the summaries and write them out
        merged = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter(handle + '.summaries/', sess.graph)

        self._create_functions()

        train_idx, valid_idx, test_idx = self._split_dataset(len(x_data), holdout, validate)

        if isinstance(x_data, np.ndarray):
            train_set_x = x_data.take(train_idx, axis=0)
            valid_set_x = x_data.take(valid_idx, axis=0)
            test_set_x = x_data.take(test_idx, axis=0)
        else:
            train_set_x = [x_data[i] for i in train_idx]
            valid_set_x = [x_data[i] for i in valid_idx]
            test_set_x = [x_data[i] for i in test_idx]

        if isinstance(y_data, np.ndarray):
            train_set_y = y_data.take(train_idx, axis=0)
            valid_set_y = y_data.take(valid_idx, axis=0)
            test_set_y = y_data.take(test_idx, axis=0)
        else:
            train_set_y = [y_data[i] for i in train_idx]
            valid_set_y = [y_data[i] for i in valid_idx]
            test_set_y = [y_data[i] for i in test_idx]

        if self.classification:
            msg = 'Distribution of Examples per set'
            print(msg)
            print('-' * len(msg))
            classes = ['Class ' + str(i) for i in range(len(np.unique(y_data)))]
            counts = dict()
            _, c = np.unique(train_set_y, return_counts=True)
            counts['train'] = c
            _, c = np.unique(valid_set_y, return_counts=True)
            counts['valid'] = c
            _, c = np.unique(test_set_y, return_counts=True)
            counts['test'] = c

            print(tabulate([['Set'] + classes + ['Total'],
                            ['Train'] + counts['train'].tolist() + [counts['train'].sum()],
                            ['Valid'] + counts['valid'].tolist() + [counts['valid'].sum()],
                            ['Test'] + counts['test'].tolist() + [counts['test'].sum()]],
                           stralign='center',
                           headers="firstrow"))
        else:
            print('Train:{0}, Valid:{1}, Test:{2}'.format(len(train_idx), len(valid_idx), len(test_idx)))

        # Early-stopping parameters
        patience = 200 * len(train_set_x)  # look as this many examples regardless
        patience_increase = 2  # wait this times much longer when a new best is found
        improvement_threshold = 0.998  # a relative improvement of this much is considered significant
        best_validation_loss = np.inf
        it = 0
        done_looping = False

        epochs = min(epochs, self.max_epochs)
        epoch = 0
        start_time = time.time()
        print(tabulate([['Epoch', 'Train Loss', 'Train Acc', 'Val Loss', 'Val Acc', 'Time']], stralign='center',
                       headers="firstrow"))
        # We iterate over epochs:
        try:
            while epoch < epochs and not done_looping:
                epoch += 1

                # In each epoch, we do a full pass over the training data:
                train_err = []
                train_acc = []
                epoch_time = time.time()

                for X, y in self._iterate_minibatches(train_set_x, train_set_y, self.batch_size):
                    self.train_fn.run(feed_dict={'data:0': X, 'targets:0': y})
                    err, acc = sess.run(['loss:0', 'acc:0'], feed_dict={'data:0': X, 'targets:0': y})
                    train_err.append(err)
                    train_acc.append(acc)

                # After that, we do a full pass over the validation data:
                val_err = []
                val_acc = []

                for X, y in self._iterate_minibatches(valid_set_x, valid_set_y, self.batch_size):
                    err, acc = sess.run(['loss:0', 'acc:0'], feed_dict={'data:0': X, 'targets:0': y, 'keep_prob:0': 1})
                    val_err.append(err)
                    val_acc.append(acc)
                summary = sess.run(merged, feed_dict={'data:0': X, 'targets:0': y, 'keep_prob:0': 1})
                writer.add_summary(summary)
                self.train_err_mem.append(np.mean(train_err))
                self.train_acc_mem.append(100 * np.mean(train_acc))
                if validate > 0:
                    self.val_err_mem.append(np.mean(val_err))
                    self.val_acc_mem.append(100 * np.mean(val_acc))
                else:
                    self.val_err_mem.append('-')
                    self.val_acc_mem.append('-')

                print(tabulate([['Epoch', 'Train Loss', 'Train Acc', 'Val Loss', 'Val Acc', 'Time'],
                                ["{}/{}".format(epoch, epochs),
                                 self.train_err_mem[-1], self.train_acc_mem[-1],
                                 self.val_err_mem[-1], self.val_acc_mem[-1],
                                 "{:.3f}s".format(time.time() - epoch_time)]],
                               tablefmt='plain', floatfmt=".6f", stralign='center',
                               headers="firstrow").rsplit('\n', 1)[-1])

                # If we have the best validation score until now
                if self.val_err_mem[-1] < best_validation_loss:

                    # Increase patience if loss improvement is good enough
                    if self.val_err_mem[-1] < best_validation_loss * improvement_threshold:
                        patience = max(patience, it * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = self.val_err_mem[-1]

                # print(self.get_all_param_values())

                # Stop if above early stopping limit
                it = (epoch - 1) * len(train_set_x)
                if patience <= it and self.patience:
                    done_looping = True
        except KeyboardInterrupt:
            return

        print('Model trained for {0} epochs. Total time: {1:.3f}s'.format(epoch, time.time() - start_time))

        if holdout > 0:
            test_loss, test_acc, self.train_vars = self.score(test_set_x, test_set_y)
            if self.classification:
                print('Test loss: {0:.6f}\tTest accuracy: {1:.6f}'.format(test_loss, test_acc))
            else:
                print('Test loss: {0:.6f}'.format(test_loss))

    def k_fold(self, x_data, y_data, epochs=1, num_folds=10, stratify=False):
        sess = tf.get_default_session()

        self._create_functions()

        if self.classification and stratify:
            folds = StratifiedKFold(y_data, n_folds=num_folds)
        else:
            folds = KFold(len(x_data), n_folds=num_folds)

        self.fold_val_losses = np.zeros((num_folds, epochs))
        self.fold_train_losses = np.zeros((num_folds, epochs))

        foldit = 0
        for train_idx, valid_idx in folds:
            print('Starting Fold {0}'.format(foldit + 1))

            if isinstance(x_data, np.ndarray):
                train_set_x = x_data.take(train_idx, axis=0)
                valid_set_x = x_data.take(valid_idx, axis=0)
            else:
                train_set_x = [x_data[i] for i in train_idx]
                valid_set_x = [x_data[i] for i in valid_idx]

            train_set_y = y_data.take(train_idx, axis=0)
            valid_set_y = y_data.take(valid_idx, axis=0)

            if self.classification:
                msg = 'Distribution of Examples per set'
                print(msg)
                print('-' * len(msg))
                classes = ['Class ' + str(i) for i in range(len(np.unique(y_data)))]
                counts = dict()
                _, c = np.unique(train_set_y, return_counts=True)
                counts['train'] = c
                _, c = np.unique(valid_set_y, return_counts=True)
                counts['valid'] = c

                print(tabulate([['Set'] + classes + ['Total'],
                                ['Train'] + counts['train'].tolist() + [counts['train'].sum()],
                                ['Valid'] + counts['valid'].tolist() + [counts['valid'].sum()]],
                               stralign='center',
                               headers="firstrow"))
            else:
                print('Train:{0}, Valid:{1}'.format(len(train_idx), len(valid_idx)))

            # Early-stopping parameters
            patience = 50 * len(train_set_x)  # look as this many examples regardless
            patience_increase = 2  # wait this times much longer when a new best is found
            improvement_threshold = 0.998  # a relative improvement of this much is considered significant
            best_validation_loss = np.inf
            it = 0
            done_looping = False

            epochs = min(epochs, self.max_epochs)
            epoch = 0
            start_time = time.time()
            print(tabulate([['Epoch', 'Train Loss', 'Val Loss', 'Time']], stralign='center', headers="firstrow"))
            # We iterate over epochs:
            while epoch < epochs and not done_looping:
                epoch += 1

                # In each epoch, we do a full pass over the training data:
                train_err = []
                train_acc = []
                epoch_time = time.time()
                for X, y in self._iterate_minibatches(train_set_x, train_set_y, self.batch_size):
                    # print(self.f(X, y))
                    self.train_fn.run(feed_dict={'data:0': X, 'targets:0': y})
                    err, acc = sess.run(['loss:0', 'acc:0'], feed_dict={'data:0': X, 'targets:0': y})
                    train_err.append(err)
                    train_acc.append(acc)

                # After that, we do a full pass over the validation data:
                val_err = []
                val_acc = []
                for X, y in self._iterate_minibatches(valid_set_x, valid_set_y, self.batch_size):
                    err, acc = sess.run(['loss:0', 'acc:0'], feed_dict={'data:0': X, 'targets:0': y, 'keep_prob:0': 1})
                    val_err.append(err)
                    val_acc.append(acc)

                self.fold_train_losses[foldit, epoch - 1] = (np.mean(train_err))
                self.fold_val_losses[foldit, epoch - 1] = (np.mean(val_err))
                print(tabulate([['Epoch', 'Train Loss', 'Val Loss', 'Time'],
                                ["{}/{}".format(epoch, epochs),
                                 self.fold_train_losses[foldit, epoch - 1],
                                 self.fold_val_losses[foldit, epoch - 1],
                                 "{:.3f}s".format(time.time() - epoch_time)]],
                               tablefmt='plain', floatfmt=".6f", stralign='center',
                               headers="firstrow").rsplit('\n', 1)[-1])

                # If we have the best validation score until now
                if self.fold_val_losses[foldit, epoch - 1] < best_validation_loss:

                    # Increase patience if loss improvement is good enough
                    if self.fold_val_losses[foldit, epoch - 1] < best_validation_loss * improvement_threshold:
                        patience = max(patience, it * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = self.fold_val_losses[foldit, epoch - 1]

                # Stop if above early stopping limit
                it = (epoch - 1) * len(train_set_x)
                if patience <= it:
                    done_looping = True

            foldit += 1
            print(
                'Fold {2}: Model trained for {0} epochs. Total time: {1:.3f}s'.format(epochs, time.time() - start_time,
                                                                                      foldit))
            val_preds = self.predict_proba(valid_set_x)
            train_preds = self.predict_proba(train_set_x)
            self.k_fold_history['train_classes'].append(train_set_y)
            self.k_fold_history['val_classes'].append(valid_set_y)
            self.k_fold_history['train_preds'].append(train_preds)
            self.k_fold_history['val_preds'].append(val_preds)
            self.reset_all_param_values()

    def score(self, X, y):
        sess = tf.get_default_session()

        test_err = []
        test_acc = []
        for X, y in self._iterate_minibatches(X, y, self.batch_size):
            err, acc = sess.run(['loss:0', 'acc:0'], feed_dict={'data:0': X, 'targets:0': y, 'keep_prob:0': 1})
            test_err.append(err)
            test_acc.append(acc)

        trainable_vars = sess.run(tf.trainable_variables(), feed_dict={'data:0': X, 'targets:0': y, 'keep_prob:0': 1})

        return np.mean(test_err), 100 * np.mean(test_acc), trainable_vars

    def predict_proba(self, X):
        sess = tf.get_default_session()

        # TODO: wtf is this ?
        self._create_functions()
        preds = []
        for X, y in self._iterate_minibatches(X, X, self.batch_size):
            preds.append(np.vstack(sess.run(self.network, feed_dict={'data:0': X, 'keep_prob:0': 1})))

        return np.vstack(preds).squeeze()

    def predict(self, X):
        if not self.classification:
            return self.predict_proba(X)
        else:
            y_pred = np.argmax(self.predict_proba(X), axis=1)
            return y_pred

    def get_num_params(self):
        num_params = 0
        for var in tf.trainable_variables():
            vec = tf.reshape(var, [-1])
            num_params += vec.get_shape()[0].value
        return num_params
        """return np.sum(
            [reduce(np.multiply, param.get_value().shape) for param in
             tnf.trainable_variables() if param])"""

    def display_network_info(self):

        print("Neural Network has {0} trainable parameters".format(self.get_num_params()))

        layers = self.net.layers

        names = []
        num_layers = 0
        shapes = []

        for key, layer in layers.items():
            if isinstance(layer, OrderedDict) or isinstance(layer, dict):
                num_layers += len(layer)
                names.append(layer)
                shapes += ['x'.join(map(str, sub_layer.get_shape().as_list()[1:])) for sub_layer in layer.values()]
            else:
                num_layers += 1
                names.append(key)
                shapes += layer.get_shape().as_list()[1:]

        ids = list(range(num_layers))

        tabula = OrderedDict([('#', ids), ('Name', names), ('Shape', shapes)])

        print(tabulate(tabula, 'keys'))

    def get_all_layers(self):
        return self.net

    def get_all_params(self):
        return tf.trainable_variables()

    def get_all_param_values(self):
        sess = tf.get_default_session()

        variables = tf.trainable_variables()
        variables_list = sess.run(variables)

        return variables_list

    def set_all_param_values(self, source):
        self._create_functions()

        sess = tf.get_default_session()

        saver = tf.train.Saver()

        saver.restore(sess, source)

    def reset_all_param_values(self):
        tf.reset_default_graph()

    def save_model(self, filename):
        sess = tf.get_default_session()

        params_filename = filename + '.model'
        saver_filename = filename + '.saver'

        params = self.train_vars
        np.savez_compressed(params_filename, *params)

        saver = tf.train.Saver()
        save_path = saver.save(sess, saver_filename)

        return 'Params saved in:' + params_filename + '/n Model saved in:' + saver_filename

    def save_train_history(self, filename):
        assert (not self.train_err_mem == [])
        filename_ = filename + '.history'
        np.savez_compressed(filename_,
                            train_err_mem=self.train_err_mem, val_err_mem=self.val_err_mem,
                            train_acc_mem=self.train_acc_mem, val_acc_mem=self.val_acc_mem)

    def save_kfold_history(self, filename):
        assert (not self.fold_train_losses[0, 0] == .0)
        filename_ = filename + '.k_fold'
        np.savez_compressed(filename_,
                            fold_train_losses=self.fold_train_losses,
                            fold_val_losses=self.fold_val_losses,
                            val_pred=self.k_fold_history['val_preds'],
                            train_pred=self.k_fold_history['train_preds'],
                            val_class=self.k_fold_history['val_classes'],
                            train_class=self.k_fold_history['train_classes'])



