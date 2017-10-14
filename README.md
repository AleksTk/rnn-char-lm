# LSTM-based Character-level Language Model

Tensorflow implementation of a lstm-based character-level language model.
Given a list of items of interest, such as baby names or place names,
the model learns to reproduce text alike.

## Usage

NB! For better performance, run this code on GPU.

### Model Training

$ python train.py [options]

```
Options:
  -h, --help            show this help message and exit
  -t TRAIN, --train=TRAIN
                        Train set location
  -m MODEL_DIR, --model_dir=MODEL_DIR
                        Directory to store trained model.
  -o OPTIMIZER, --optimizer=OPTIMIZER
                        Objective function optimiser ('rmsprop', 'adam' or
                        'gd' for gradient descent)
  -n NUM_EPOCHS, --num_epochs=NUM_EPOCHS
                        Number of passes over training dataset
  -b BATCH_SIZE, --batch_size=BATCH_SIZE
                        Batch size
  -e EMBEDDING_SIZE, --embedding_size=EMBEDDING_SIZE
                        Character embedding size
  -r HIDDEN_SIZE, --hidden_size=HIDDEN_SIZE
                        Rnn hidden layer size
  -l LEARNING_RATE, --learning_rate=LEARNING_RATE
                        Initial learning rate
  -d DECAY_RATE, --decay_rate=DECAY_RATE
                        Learning rate decay
  -D DROPOUT, --dropout=DROPOUT
                        Dropout on the LSTM output (0 = no dropout)
  -L LOG_STEP, --log_step=LOG_STEP
                        Print progress after n batches are processed.
```

For example, to train a model using data `data/uk_towns.txt` with 10 epochs and 0.5 dropout on outputs, run

```
$ python train.py -t data/uk_towns.txt -m uk_towns -n 10 -D 0.5

+------------------+------------------+------------------+----------------------------------------------------------------------------------+
|            epoch |       train-loss |   train-accuracy | examples                                                                         |
|------------------+------------------+------------------+----------------------------------------------------------------------------------|
|                0 |         4.419108 |         0.013193 | T2qCf³B-0w,gNƒBL€Taƒmy2oj, PGcn')PjjpXÃL3Y€K,X,fÂfi&N&)9IFKƒZwx0Ws²xRkA OnB,)... |
|                0 |         4.318801 |         0.120107 | ArxGCG³eVaVAwweln,WHgqfMYCh5hk-&YâeOeoeaha,o7º.€p- ^ox3vade_n,6DVX¹Mg63caVkºa... |
...
|                9 |         1.928021 |         0.429141 | Bintstone Chorpe,Llanfinabh,Saxtingham,Julsey,Babgeich,Pontforsha,Miltnow End... |
|                9 |         1.943015 |         0.427479 | Manton End,Stenhall,Nydside,Nettington Meithworrie,Winshall,Rosswood,Kil Mait... |
|                9 |         1.970449 |         0.419188 | Rhede,Eccoor,Crowton,Nont Heaed,Faven's Hill,Cufkerton,Beckkyldworth,Littling... |

```

### Generating text

$ python generate.py [options]

```
Options:
  -h, --help            show this help message and exit
  -m MODEL_DIR, --model_dir=MODEL_DIR
                        Model directory
  -t TEMPERATURE, --temperature=TEMPERATURE
                        Temperature (0 > t <= 1.0). Large temperature implies
                        more randomness.
  -n NUM_WORDS, --num_words=NUM_WORDS
                        Number of items to generate
  -p PREFIX, --prefix=PREFIX
                        Word prefix

```

For example, to generate 5 items with temperature 1 using pre-trained model `uk_towns/epoch9`, run

```
$ python generate.py -m uk_towns/epoch9 -t 1 -n 5

Heuston
Wallboty
Headbernis
Old Lane
Daive
```

##  Requirements

```
python 3.4
tensorflow 1.2.1
```

## Install

```
pip install -r requirements.txt
```

## License

MIT
