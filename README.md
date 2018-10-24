# deepchess

**What is it ?**

A Chess engine trained using unsupervised ( stacked autoencoders ) + supervsed training ( feed-forward ). For further details read this [ paper](http://www.cs.tau.ac.il/~wolf/papers/deepchess.pdf) 

**What it does best ?**

Compare two chess positions and output the better position ( from white's perspective, the other position will be better for black.)

**What it can do ?**

Can play a whole game of chess but evaluation is very slow on CPUs therefore needs refinements

**Future ?**

Add refinements for faster evaluation.

## Dependencies:

[Tensorflow](https://www.tensorflow.org/)

[Python 3.5.x](https://www.python.org/downloads/release/python-350/)

[ python-chess 0.17.0 ](http://python-chess.readthedocs.io/en/v0.17.0/ )

[ tf-deploy ]( https://github.com/riga/tfdeploy )

## Data

### Chess games were acquired from these sites

**PGN Mentor** - selected candidates and opening http://www.pgnmentor.com/files.html#players

**CCRL** (Computer Chess Rating Lists) website.( http://www.computerchess.org.uk/ccrl/4040/games.html )


This is the data that has been converted to numpy array format and stored in gzip format. This is the data on which the model is trained and tested.

https://drive.google.com/open?id=0B_g0vyfZ_jwGVkRDU002M0lZX2c

## The Computational Graph

![alt text](https://github.com/vajjhala/ChessBot/blob/master/graph.png)

## Usage

Download all the files and  compare two moves using the `better(_ , _)` function in`evaluator.py` file 

``` python
m1 = chess.Board('rn1qkbnr/4p1pp/p7/1B2N3/1Q1P4/1P4P1/2P2P1P/R1B1K2R w KQkq - 0 10') # FEN of move 1
m2 = chess.Board('rn1qkbnr/4p1pp/p7/1Q2N3/3P4/1P4P1/2P2P1P/R1B1KB1R w KQkq - 0 10') # FEN of move 2
better(m1, m2)
```
