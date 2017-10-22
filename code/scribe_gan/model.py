import numpy as np
import numpy.matlib
import math
import random
import os
import xml.etree.ElementTree as ET

import tensorflow as tf
from utils import *

class Model():
    def __init__(self, args, logger):
        self.logger = logger

        # ----- transfer some of the args params over to the model

        # model params
        self.rnn_size = args.rnn_size
        self.train = args.train
        self.nmixtures = args.nmixtures
        self.kmixtures = args.kmixtures
        self.batch_size = args.batch_size if self.train else 1 # training/sampling specific
        self.tsteps = args.tsteps if self.train else 1 # training/sampling specific
        self.alphabet = args.alphabet
        self.d_layers = args.d_layers
        # training params
        self.dropout = args.dropout
        self.grad_clip = args.grad_clip
        # misc
        self.tsteps_per_ascii = args.tsteps_per_ascii
        self.data_dir = args.data_dir

        self.graves_initializer = tf.truncated_normal_initializer(mean=0., stddev=.075, seed=None, dtype=tf.float32)
        self.window_b_initializer = tf.truncated_normal_initializer(mean=-3.0, stddev=.25, seed=None, dtype=tf.float32) # hacky initialization

        self.logger.write('\tusing alphabet{}'.format(self.alphabet))
        self.char_vec_len = len(self.alphabet) + 1 #plus one for <UNK> token
        self.ascii_steps = args.tsteps//args.tsteps_per_ascii


        # ----- build the basic recurrent network architecture
        cell_func = tf.contrib.rnn.LSTMCell # could be GRUCell or RNNCell
        self.cell0 = cell_func(args.rnn_size, state_is_tuple=True, initializer=self.graves_initializer)
        self.cell1 = cell_func(args.rnn_size, state_is_tuple=True, initializer=self.graves_initializer)
        self.cell2 = cell_func(args.rnn_size, state_is_tuple=True, initializer=self.graves_initializer)

        if (self.train and self.dropout < 1): # training mode
                self.cell0 = tf.contrib.rnn.DropoutWrapper(self.cell0, output_keep_prob = self.dropout)
                self.cell1 = tf.contrib.rnn.DropoutWrapper(self.cell1, output_keep_prob = self.dropout)
                self.cell2 = tf.contrib.rnn.DropoutWrapper(self.cell2, output_keep_prob = self.dropout)

        self.input_data = tf.placeholder(dtype=tf.float32, shape=[None, self.tsteps, 3], name='input')
        self.target_data = tf.placeholder(dtype=tf.float32, shape=[None, self.tsteps, 3], name='target')
        self.istate_cell0 = self.cell0.zero_state(batch_size=self.batch_size, dtype=tf.float32)
        self.istate_cell1 = self.cell1.zero_state(batch_size=self.batch_size, dtype=tf.float32)
        self.istate_cell2 = self.cell2.zero_state(batch_size=self.batch_size, dtype=tf.float32)

        #slice the input volume into separate vols for each tstep
        inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(self.input_data, self.tsteps, 1)]
        #build cell0 computational graph
        outs_cell0, self.fstate_cell0 = tf.contrib.legacy_seq2seq.rnn_decoder(inputs, self.istate_cell0, self.cell0, loop_function=None, scope='cell0')


# ----- build the gaussian character window
        def get_window(alpha, beta, kappa, c):
            # phi -> [? x 1 x ascii_steps] and is a tf matrix
            # c -> [? x ascii_steps x alphabet] and is a tf matrix
            ascii_steps = c.get_shape()[1].value #number of items in sequence
            phi = get_phi(ascii_steps, alpha, beta, kappa)
            #phishape = tf.shape(phi)
            #chsape = tf.shape(c)
            #phi = tf.Print(phi, [phishape, chsape])
            window = tf.matmul(phi,c)
            window = tf.squeeze(window, [1]) # window ~ [?,alphabet]
            return window, phi

        #get phi for all t,u (returns a [1 x tsteps] matrix) that defines the window
        def get_phi(ascii_steps, alpha, beta, kappa):
            # alpha, beta, kappa -> [?,kmixtures,1] and each is a tf variable
            u = np.linspace(0,ascii_steps-1,ascii_steps) # weight all the U items in the sequence
            kappa_term = tf.square( tf.subtract(kappa,u))
            exp_term = tf.multiply(-beta,kappa_term)
            phi_k = tf.multiply(alpha, tf.exp(exp_term))
            phi = tf.reduce_sum(phi_k,1, keep_dims=True)
            return phi # phi ~ [?,1,ascii_steps]

        def get_window_params(i, out_cell0, kmixtures, prev_kappa, reuse=True):
            hidden = out_cell0.get_shape()[1]
            n_out = 3*kmixtures
            with tf.variable_scope('window',reuse=reuse):
                    window_w = tf.get_variable("window_w", [hidden, n_out], initializer=self.graves_initializer)
                    window_b = tf.get_variable("window_b", [n_out], initializer=self.window_b_initializer)
            abk_hats = tf.nn.xw_plus_b(out_cell0, window_w, window_b) # abk_hats ~ [?,n_out]
            abk = tf.exp(tf.reshape(abk_hats, [-1, 3*kmixtures,1])) # abk_hats ~ [?,n_out] = "alpha, beta, kappa hats"

            alpha, beta, kappa = tf.split(abk, 3, 1) # alpha_hat, etc ~ [?,kmixtures]
            kappa = kappa + prev_kappa
            return alpha, beta, kappa # each ~ [?,kmixtures,1]

        self.init_kappa_d = tf.placeholder(dtype=tf.float32, shape=[None, self.kmixtures, 1], name='kappa_d')
        self.init_kappa_g = tf.placeholder(dtype=tf.float32, shape=[None, self.kmixtures, 1], name='kappa_g')
        self.char_seq = tf.placeholder(dtype=tf.float32, shape=[None, self.ascii_steps, self.char_vec_len], name='char_seq')
        prev_kappa = self.init_kappa_g
        prev_window = self.char_seq[:,0,:]

        #add gaussian window result
        reuse = False
        for i in range(len(outs_cell0)):
            [alpha, beta, new_kappa] = get_window_params(i, outs_cell0[i], self.kmixtures, prev_kappa, reuse=reuse)
            outs_cell0[i] = tf.verify_tensor_all_finite(outs_cell0[i], 'c0i')
            window, phi = get_window(alpha, beta, new_kappa, self.char_seq)
            outs_cell0[i] = tf.concat((outs_cell0[i],window), 1) #concat outputs
            outs_cell0[i] = tf.concat((outs_cell0[i],inputs[i]), 1) #concat input data
            prev_kappa = new_kappa
            prev_window = window
            reuse = True

        #save some attention mechanism params (useful for sampling/debugging later)
        self.window = window
        self.phi = phi
        self.new_kappa_g = new_kappa
        self.alpha = alpha

        # ----- finish building LSTMs 2 and 3
        outs_cell1, self.fstate_cell1 = tf.contrib.legacy_seq2seq.rnn_decoder(outs_cell0, self.istate_cell1, self.cell1, loop_function=None, scope='cell1')

        outs_cell2, self.fstate_cell2 = tf.contrib.legacy_seq2seq.rnn_decoder(outs_cell1, self.istate_cell2, self.cell2, loop_function=None, scope='cell2')

        # x1 and x2
        n_out = 3
        with tf.variable_scope('lstm_to_coord'):
            coord_w = tf.get_variable("output_w", [self.rnn_size, n_out], initializer=self.graves_initializer)
            coord_b = tf.get_variable("output_b", [n_out], initializer=self.graves_initializer)

        out_cell2 = tf.reshape(tf.concat(outs_cell2, 1), [-1, args.rnn_size]) #concat outputs for efficiency
        self.output_gen = tf.nn.xw_plus_b(out_cell2, coord_w, coord_b)

        def discriminator(input_data):
            #slice the input volume into separate vols for each tstep

            dcell0 = tf.contrib.rnn.LSTMCell(args.rnn_size, state_is_tuple=True, initializer=self.graves_initializer)

            #inputs = [tf.squeeze(input_, [0]) for input_ in tf.split(input_data, self.tsteps, 0)]
            #dcell1_size = args.rnn_size + self.char_vec_len + int(inputs[0].shape[-1])
            #dcell = tf.contrib.rnn.LSTMCell(dcell1_size, state_is_tuple=True, initializer=self.graves_initializer)

            inputs = [tf.squeeze(input_, [0]) for input_ in tf.split(input_data, self.tsteps, 0)]
            dcell1_size = args.rnn_size + self.char_vec_len + int(inputs[0].shape[-1])
            dcell = tf.contrib.rnn.LSTMCell(dcell1_size, state_is_tuple=True, initializer=self.graves_initializer)

            if (self.train and self.dropout < 1): # training mode
                dcell0 = tf.contrib.rnn.DropoutWrapper(dcell0, output_keep_prob = self.dropout)
                dcell = tf.contrib.rnn.DropoutWrapper(dcell, output_keep_prob = self.dropout)

            self.istate_dcell0 = dcell0.zero_state(batch_size=self.batch_size, dtype=tf.float32)

            #build dcell0 computational graph
            outs_dcell0, self.fstate_dcell0 = tf.nn.dynamic_rnn(dcell0, input_data, initial_state=self.istate_dcell0, time_major=True)

            outs_dcell0 = [tf.squeeze(output_, [0]) for output_ in tf.split(outs_dcell0, self.tsteps, 0)]

            #add gaussian window result
            reuse = False
            prev_kappa = self.init_kappa_d
            for i in range(self.tsteps):
                [alpha, beta, new_kappa] = get_window_params(i, outs_dcell0[i], self.kmixtures, prev_kappa, reuse=reuse)
                window, phi = get_window(alpha, beta, new_kappa, self.char_seq)
                outs_dcell0[i] = tf.concat((outs_dcell0[i],window), 1) #concat outputs
                outs_dcell0[i] = tf.concat((outs_dcell0[i],inputs[i]), 1) #concat input data
                prev_kappa = new_kappa
                prev_window = window
                reuse = True

            self.new_kappa_d = new_kappa

            outs_dcell0 = tf.stack(outs_dcell0, 0)

            #stacked_dcell = tf.nn.rnn_cell.MultiRNNCell([dcell for _ in range(self.d_layers)], state_is_tuple=True)
            #self.istate_dcell1 = stacked_dcell.zero_state(self.batch_size, tf.float32)
            #dcell_outputs, self.fstate_dcell1 = tf.nn.dynamic_rnn(stacked_dcell, outs_dcell0, initial_state=self.istate_dcell1, time_major=True)

            self.istate_dcell1 = dcell.zero_state(self.batch_size, tf.float32)
            dcell_outputs, self.fstate_dcell1 = tf.nn.dynamic_rnn(dcell, outs_dcell0, initial_state=self.istate_dcell1, time_major=True, scope='cell1')


            n_out = 1
            with tf.variable_scope('coord_discriminator'):
                coord_w = tf.get_variable("output_w", [dcell1_size, n_out], initializer=self.graves_initializer)
                coord_b = tf.get_variable("output_b", [n_out], initializer=self.graves_initializer)

            dcell_outputs = tf.reshape(tf.concat(dcell_outputs, 1), [-1, dcell1_size]) #concat outputs for efficiency
            prob = tf.sigmoid(tf.nn.xw_plus_b(dcell_outputs, coord_w, coord_b))
            #prob = tf.Print(prob, [prob])
            return prob

        with tf.variable_scope('D') as scope:
            output_data = tf.reshape(self.output_gen, [self.batch_size, self.tsteps, 3])
            #input_gen = tf.concat([self.input_data, output_data], 2)
            input_gen = output_data
            input_gen = tf.transpose(input_gen, perm=[1, 0, 2])
            #input_gen = tf.Print(input_gen, [input_gen], message='gen', summarize=18)
            #input_gen = tf.Print(input_gen, [input_gen[0][0], input_gen[1][0], input_gen[2][0]], message='gen', summarize=18)
            self.d_gen = discriminator(input_gen)
            #self.d_gen = tf.Print(self.d_gen, [self.d_gen], message='gen')
            #self.d_gen = tf.verify_tensor_all_finite(self.d_gen, 'dgen')

            scope.reuse_variables()
            
            #input_real = tf.concat([self.input_data, self.target_data], 2)
            input_real = self.input_data
            input_real = tf.transpose(input_real, perm=[1, 0, 2])
            #input_real = tf.Print(input_real, [input_real[0], input_real[1], input_real[2]], message='real', summarize=1000)
            self.d_real = discriminator(input_real)
            #self.d_real = tf.Print(self.d_real, [self.d_real], message='real')

        # reshape target data (as we did the input data)

        d_loss = tf.reduce_mean(-tf.log(tf.clip_by_value(self.d_gen, 1e-5, 1.0)) \
                -tf.log(tf.clip_by_value((1 - self.d_real), 1e-5, 1.0)))

        self.cost_d = d_loss# / (self.batch_size * self.tsteps)


        g_loss = tf.clip_by_value((1 - self.d_gen), 1e-5, 1.0)
        g_loss = -tf.log(g_loss)
        g_loss = tf.verify_tensor_all_finite(g_loss, 'gloss_log')
        g_loss = tf.reduce_mean(g_loss)
        #g_loss = tf.verify_tensor_all_finite(g_loss, 'gloss_mean')
        self.cost_g = g_loss# / (self.batch_size * self.tsteps)

        #loss = get_loss(self.pi, x1_data, x2_data, eos_data, self.mu1, self.mu2, self.sigma1, self.sigma2, self.rho, self.eos)
        #self.cost = loss / (self.batch_size * self.tsteps)

        # ----- bring together all variables and prepare for training
        self.learning_rate = tf.Variable(0.0, trainable=False)
        self.decay = tf.Variable(0.0, trainable=False)
        self.momentum = tf.Variable(0.0, trainable=False)

        tvars_d = [v for v in tf.trainable_variables() if v.name.startswith('D/')]
        tvars_g = [v for v in tf.trainable_variables() if not v.name.startswith('D/')]
        d_grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost_d, tvars_d), self.grad_clip)
        g_grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost_g, tvars_g), self.grad_clip)

        if args.optimizer == 'adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        elif args.optimizer == 'rmsprop':
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, decay=self.decay, momentum=self.momentum)
        else:
            raise ValueError("Optimizer type not recognized")

        self.train_op_d = self.optimizer.apply_gradients(zip(d_grads, tvars_d))
        self.train_op_g = self.optimizer.apply_gradients(zip(g_grads, tvars_g))

        # ----- some TensorFlow I/O
        self.sess = tf.InteractiveSession()
        self.saver = tf.train.Saver(tf.global_variables())
        self.sess.run(tf.global_variables_initializer())


    # ----- for restoring previous models
    def try_load_model(self, save_path):
        load_was_success = True # yes, I'm being optimistic
        global_step = 0
        try:
            save_dir = '/'.join(save_path.split('/')[:-1])
            ckpt = tf.train.get_checkpoint_state(save_dir)
            load_path = ckpt.model_checkpoint_path
            self.saver.restore(self.sess, load_path)
        except:
            self.logger.write("no saved model to load. starting new session")
            load_was_success = False
        else:
            self.logger.write("loaded model: {}".format(load_path))
            self.saver = tf.train.Saver(tf.global_variables())
            global_step = int(load_path.split('-')[-1])
        return load_was_success, global_step
