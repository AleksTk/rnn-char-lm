"""
Trains LSTM-based character-level language model.

Usage: python train.py -h
"""
import os
import sys
import optparse
import random

from table_logger import TableLogger

from model import Model

optparser = optparse.OptionParser()
optparser.add_option(
    "-t", "--train", default="",
    help="Train set location"
)
optparser.add_option(
    "-m", "--model_dir", default="",
    help="Directory to store trained model."
)
optparser.add_option(
    "-o", "--optimizer", default="adam",
    help="Objective function optimiser ('rmsprop', 'adam' or 'gd' for gradient descent)"
)
optparser.add_option(
    "-n", "--num_epochs", default="50",
    type='int', help="Number of passes over training dataset"
)
optparser.add_option(
    "-b", "--batch_size", default="30",
    type='int', help="Batch size"
)
optparser.add_option(
    "-e", "--embedding_size", default="50",
    type='int', help="Character embedding size"
)
optparser.add_option(
    "-r", "--hidden_size", default="100",
    type='int', help="Rnn hidden layer size"
)
optparser.add_option(
    "-l", "--learning_rate", default="0.015",
    type='float', help="Initial learning rate"
)
optparser.add_option(
    "-d", "--decay_rate", default="0.0",
    type='float', help="Learning rate decay"
)
optparser.add_option(
    "-D", "--dropout", default="0.5",
    type='float', help="Dropout on the LSTM output (0 = no dropout)"
)
optparser.add_option(
    "-L", "--log_step", default="10",
    type='int', help="Print progress after n batches are processed."
)

opts = optparser.parse_args()[0]

# Check parameters validity
try:
    assert opts.train != "", "Train set location [-t] not specified"
    assert os.path.isfile(opts.train), "Train set '{}' not found".format(opts.train)
    assert opts.model_dir != "", "Model output directory [-m] not specified"
    assert opts.optimizer in ("adam", "rmsprop", "gd"), "Invalid optimiser {}".format(opts.optimizer)
    assert opts.num_epochs > 0
    assert 0. <= opts.dropout < 1.0
    assert opts.embedding_size > 0
    assert opts.hidden_size > 0
except AssertionError as e:
    print('AssertionError:', e)
    optparser.print_help()
    sys.exit()

# load train set
words = [w.rstrip() for w in open(opts.train, encoding='utf-8')]
vocab = list(set(c for w in words for c in w))  # list of unique characters.

model = Model(vocab=vocab,
              embedding_size=opts.embedding_size,
              decay_rate=opts.decay_rate,
              dropout=opts.dropout,
              learning_rate=opts.learning_rate,
              hidden_size=opts.hidden_size,
              optimizer=opts.optimizer,
              pad_char='_')

# setup logging
tbl = TableLogger(columns='epoch,train-loss,train-accuracy,examples',
                  rownum=True,
                  time_delta=True,
                  default_colwidth=16,
                  colwidth={'examples': 80})

print("Train file:", opts.train)
print("Output model directory:", opts.model_dir)
print("Character embedding size:", opts.embedding_size)
print("Vocabulary ({} characters): {}".format(len(vocab), ''.join(vocab)))
print("Epochs to train:", opts.num_epochs)
print("Batch size:", opts.batch_size)
print("Initial learning rate:", opts.learning_rate)
print("Learning rate decay:", opts.decay_rate)
print("Dropout:", opts.dropout)
print("Optimizer:", opts.optimizer)
print("Log step:", opts.log_step)

# train loop
for e in range(opts.num_epochs):
    random.shuffle(words)
    for i in range(0, len(words), opts.batch_size):
        words_batch = words[i:i + opts.batch_size]
        mean_loss, mean_acc = model.train(words_batch, e)
        if i % opts.log_step == 0:
            examples = ','.join([model.generate() for _ in range(10)])
            tbl(e, mean_loss, mean_acc, examples)

    model.save(os.path.join(opts.model_dir, "epoch" + str(e)))

model.close()
