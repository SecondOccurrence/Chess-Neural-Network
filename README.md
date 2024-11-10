This was made as part of a final assignment for a Machine Learning and Artificial Intelligence unit at University

# Project Description:

This project uses a Neural Network to predict the next 'best' possible move, given a board state, in a game of Chess.

To predict the next best move in a chess game, the model outputs a board score for the current state of the board. It then calculates the scores for the state of the board after each possible legal move is applied. The most positive or least negative change in board score will indicate the best move to make.

# Provided Functionalities

- Playing against the model in a game of chess in the command line
- Training and testing the model

## Playing Against the Model

- `python3 -m src.play_game`
- The model is loaded using the weight file found at `./saves/chess_nn_weights.pth`

This allows you to play out a game of chess, in the command line, against the model.

## Training and Testing the Model

**For Training:**
- `python3 -m src.training.train`

**For Testing:**
- `python3 -m src.training.test`

Both require a dataset. This has been trained using a custom dataset of 100K samples. To generate this yourself, use `src/data/generate.py`. Although, this would take a while.
