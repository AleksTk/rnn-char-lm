"""Tensorflow implementation of a RNN-based character-level language model."""
import os
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn


class Model(object):
    def __init__(self, vocab, embedding_size, hidden_size, dropout, learning_rate, decay_rate,
                 optimizer="adam", sos_char='^', eos_char='$', pad_char=' ', init=True):
        """
        Character-level RNN language model.

        Args:
            vocab (list): list of unique characters in training set
            embedding_size (int): character embedding size
            hidden_size (int): RNN hidden layer size
            dropout (float): dropout on the LSTM output (0 = no dropout)
            learning_rate (float): initial learning rate
            decay_rate (float): learning rate decay
                The actual learning rate at epoch `t` will be: lr = `learning_rate` / (1 + `t` * `decay_rate`)
            optimizer (str): optimization method to use.
                Options are: "adam", "rmsprop" or "gd" for gradient descent.
            sos_char (str): start-of-sequence character
            eos_char (str): end-of-sequence character
            pad_char (str): character used for padding
            init (bool): initialize global variables in the current session.
                Set to False when restoring saved model.
        """
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.optimizer = optimizer
        self.dropout = dropout
        self.sos_char = sos_char
        self.eos_char = eos_char
        self.pad_char = pad_char

        # process vocabulary
        self.vocab = [self.pad_char, self.eos_char, self.sos_char] + vocab
        assert len(self.vocab) == len(set(self.vocab)), "Vocabulary contains special characters."
        self.vocab_size = len(self.vocab)
        self.char2idx = {c: i for i, c in enumerate(self.vocab)}
        self.idx2char = {i: c for i, c in enumerate(self.vocab)}

        # Define computational graph

        # Placeholders
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        seq_len = tf.placeholder(tf.int32, shape=(None,), name='L')
        x = tf.placeholder(tf.int64, shape=(None, None), name='X')
        y = tf.placeholder(tf.int64, shape=(None, None), name='Y')
        dropout = tf.placeholder(tf.float32)
        batch_size = tf.placeholder(tf.int32)
        temperature = tf.placeholder(tf.float32)

        # Embedding layer
        embeddings = tf.get_variable('embedding_matrix', [self.vocab_size, embedding_size])
        rnn_inputs = tf.nn.embedding_lookup(embeddings, x, name="embeddings")

        # RNN
        cell = rnn.BasicLSTMCell(hidden_size)
        initial_state = cell.zero_state(tf.shape(x)[0], tf.float32)
        rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, sequence_length=seq_len,
                                                     initial_state=initial_state, dtype=tf.float32)

        # add dropout to rnn outputs
        rnn_outputs = tf.nn.dropout(rnn_outputs, 1. - dropout)

        # flatten data
        rnn_outputs_flat = tf.reshape(rnn_outputs, [-1, hidden_size])
        y_flat = tf.reshape(y, [-1], name="y_flat")

        # softmax layer
        with tf.variable_scope('softmax'):
            W = tf.get_variable('W', [hidden_size, self.vocab_size])
            b = tf.get_variable('b', [self.vocab_size], initializer=tf.constant_initializer(0.0))

        # get predictions
        logits = tf.matmul(rnn_outputs_flat, W) + b
        preds = tf.nn.softmax(logits)
        preds_temp = tf.nn.softmax(tf.log(preds) / temperature)

        # Calculate loss
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_flat, name='losses')
        # Mask the losses
        mask = tf.sign(tf.to_float(y_flat))
        masked_losses = mask * losses
        # Bring back to [batch, time] shape
        masked_losses = tf.reshape(masked_losses, tf.shape(y))
        # Calculate mean loss
        mean_loss_by_example = tf.reduce_sum(masked_losses, reduction_indices=1) / tf.to_float(seq_len)
        mean_loss = tf.reduce_mean(mean_loss_by_example)

        # Calculate per character accuracy
        # Count padding-chars in y
        npad = tf.to_float(tf.shape(y_flat, out_type=tf.int64)[0] - tf.count_nonzero(y_flat))
        # Get best predictions
        preds_flat = tf.argmax(preds, 1)
        # Make sure padding-characters are zeros
        preds_flat = tf.to_float(preds_flat) * mask
        # Count correct predictions excluding padding characters
        correct_preds = tf.equal(tf.to_int64(preds_flat), y_flat)
        correct_preds = tf.reduce_sum(tf.to_float(correct_preds)) - npad
        # Count all predictions excluding padding characters
        all_preds = tf.to_float(tf.shape(preds_flat)[0]) - npad
        accuracy = correct_preds / all_preds

        if self.optimizer == "rmsprop":
            optimizer = tf.train.RMSPropOptimizer
        elif self.optimizer == "adam":
            optimizer = tf.train.AdamOptimizer
        elif self.optimizer == "gd":
            optimizer = tf.train.GradientDescentOptimizer
        else:
            raise ValueError("Invalid optimiser {}".format(self.optimizer))

        # optimise
        train_step = optimizer(learning_rate=learning_rate).minimize(mean_loss)

        # Graph entry points
        self._x = x
        self._y = y
        self._seq_len = seq_len
        self._batch_size = batch_size
        self._lr = learning_rate
        self._dropout = dropout
        self._temperature = temperature
        self._preds = preds
        self._preds_temp = preds_temp
        self._loss = mean_loss
        self._accuracy = accuracy
        self._train_step = train_step
        self._initial_state = initial_state
        self._final_state = final_state

        self._session = tf.Session()

        if init is True:
            self._session.run(tf.global_variables_initializer())

    def train(self, words, epoch):
        """
        Trains a model given a batch of words.

        Args:
            words (list of str): batch of words
            epoch (int): current epoch

        Returns:
            tuple of (training loss, training accuracy)
        """
        X, Y, seq_len = self._words2batch(words)
        initial_state = (np.zeros((X.shape[0], self.hidden_size)),
                         np.zeros((X.shape[0], self.hidden_size)))
        feed_in = {self._x: X,
                   self._y: Y,
                   self._seq_len: seq_len,
                   self._batch_size: X.shape[0],
                   self._dropout: self.dropout,
                   self._initial_state: initial_state,
                   self._lr: self.learning_rate / (1. + epoch * self.decay_rate),
                   }
        feed_out = [self._loss,
                    self._accuracy,
                    self._train_step]
        loss, acc, _ = self._session.run(feed_out, feed_in)
        return loss, acc

    def generate(self, prefix="", temperature=1.0):
        """
        Generates one word.

        Args:
            prefix (str): optional word prefix to use as a seed.
            temperature (float): Temperature in range [0; 1). Larger temperature = more randomness. Defaults to 1.

        Returns:
            str: generated word

        """
        assert 0 < temperature <= 1.0
        state = (np.zeros((1, self.hidden_size)),
                 np.zeros((1, self.hidden_size)))
        word = prefix
        if not word.startswith(self.sos_char):
            word = self.sos_char + word
        c = word
        while word[-1] != self.eos_char and len(word) < 30:
            feed_in = {self._x: np.array([self._word2vec(c)]),
                       self._seq_len: np.array([len(c)]),
                       self._dropout: 0.0,
                       self._batch_size: 1,
                       self._temperature: temperature,
                       self._initial_state: state}
            feed_out = [self._preds_temp,
                        self._final_state]
            preds, state = self._session.run(feed_out, feed_in)
            # last character distribution
            preds = preds[-1]
            char_idx = np.random.choice(len(preds), p=preds)
            c = self.idx2char[char_idx]
            word += c
        return word[1:-1]

    def save(self, model_dir):
        """Saves model to directory `model_dir`."""
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # pickle model state
        state = {
            'vocab': self.vocab[3:],  # skip special characters
            'optimizer': self.optimizer,
            'embedding_size': self.embedding_size,
            'hidden_size': self.hidden_size,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'decay_rate': self.decay_rate,
            'sos_char': self.sos_char,
            'eos_char': self.eos_char,
            'pad_char': self.pad_char
        }
        with open(os.path.join(model_dir, 'model.pkl'), 'wb') as f:
            pickle.dump(state, f, protocol=3)

        # save tf session
        saver = tf.train.Saver()
        saver.save(self._session, os.path.join(model_dir, 'model.tf'))

    @staticmethod
    def restore(model_dir):
        """Restored saved model from directory `model_dir`."""
        with open(os.path.join(model_dir, 'model.pkl'), 'rb') as f:
            state = pickle.load(f)
        model = Model(init=False, **state)
        saver = tf.train.Saver()
        saver.restore(model._session, os.path.join(model_dir, 'model.tf'))
        return model

    def close(self):
        """Closes session and resets computational graph."""
        tf.reset_default_graph()
        self._session.close()

    def _words2batch(self, words):
        X, Y, seq_len = [], [], []
        max_word_len = max(len(w) for w in words) + 1
        for word in words:
            npad = max_word_len - len(word) - 1

            x = self.sos_char + word + self.pad_char * npad
            x_vec = self._word2vec(x)
            X.append(x_vec)

            y = word + self.eos_char + self.pad_char * npad
            y_vec = self._word2vec(y)
            Y.append(y_vec)

            x_len = len(word) + 1
            seq_len.append(x_len)

        return np.array(X), np.array(Y), np.array(seq_len)

    def _word2vec(self, word):
        return [self.char2idx[c] for c in word]
