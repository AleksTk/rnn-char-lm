"""
Generates text using pre-trained model.

Usage: python generate.py -h
"""
import os
import sys
import optparse

from model import Model

optparser = optparse.OptionParser()
optparser.add_option(
    "-m", "--model_dir", default="",
    help="Model directory"
)
optparser.add_option(
    "-t", "--temperature", default="1.0",
    type='float', help="Temperature (0 > t <= 1.0). Large temperature implies more randomness."
)
optparser.add_option(
    "-n", "--num_words", default="50",
    type='int', help="Number of items to generate"
)
optparser.add_option(
    "-p", "--prefix", default="",
    help="Word prefix"
)

opts = optparser.parse_args()[0]

# Check parameters validity
try:
    assert opts.model_dir != "", "Model directory [-m] nor specified"
    assert os.path.isdir(opts.model_dir), "Model directory {} not found".format(opts.model_dir)
    assert 0. < opts.temperature <= 1.0, "Temperature [-t] should be in the range (0, 1]"
    assert opts.num_words > 0
except AssertionError as e:
    print('AssertionError:', e)
    optparser.print_help()
    sys.exit()

model = Model.restore(opts.model_dir)

for _ in range(opts.num_words):
    w = model.generate(prefix=opts.prefix, temperature=opts.temperature)
    print(w)

model.close()
