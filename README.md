# LSTM-based Character-level Language Model

Tensorflow implementation of a lstm-based character-level language model.
Given a list of items of interest, such as baby names or place names,
the model learns to reproduce text alike.

## Usage

NB! For better performance, run this code on GPU.

### Training model

```
python train.py -t data/uk_towns.txt -m uk_towns -n 10 -D 0.5

+------------------+------------------+------------------+----------------------------------------------------------------------------------+
|            epoch |       train-loss |   train-accuracy | examples                                                                         |
|------------------+------------------+------------------+----------------------------------------------------------------------------------|
|                0 |         4.419108 |         0.013193 | T2qCf³B-0w,gNƒBL€Taƒmy2oj, PGcn')PjjpXÃL3Y€K,X,fÂfi&N&)9IFKƒZwx0Ws²xRkA OnB,)... |
|                0 |         4.318801 |         0.120107 | ArxGCG³eVaVAwweln,WHgqfMYCh5hk-&YâeOeoeaha,o7º.€p- ^ox3vade_n,6DVX¹Mg63caVkºa... |
...
```

For help, run

```
python train.py -h
```

### Generating text

```
python generate.py -m uk_towns/epoch9 -t 1 -n 5

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
