import numpy as np
import chess
import copy
from util import *

# function to determine which position has higher probability
def choiceMove(first, second):

	bitpos1 = valtobit(FENtoval(first.fen()))
    bitpos2 = valtobit(FENtoval(second.fen()))
    #bitpos = [bitpos1, bitpos2]
	#NEED TO CALL FUNCTION THAT GIVES PROBABILITY GIVEN TESTING POSITIONS
    ##result = y.eval({x: bitpos})
	##if result[0][0] > result [0][1]:
	##	return (first, second)
	##else:
    ##    return (second, first)

# alpha-beta pruning
def alphabeta(node, depth, alpha, beta, maximizingPlayer):
	if depth == 0:
		return node
	if maximizingPlayer:
		v = -1
		for move in node.legal_moves:
			cur = copy.copy(node)
			cur.push(move)
			if v == -1:
				v = alphabeta(cur, depth-1, alpha, beta, False) 
			if alpha == -1:
				alpha = v
		
			v = choiceMove(v, alphabeta(cur, depth-1, alpha, beta, False))[0]
			alpha = choiceMove(alpha, v)[0] 
			if beta != 1:
				if choiceMove(alpha, beta)[0] == alpha:
					break
		return v 
	else:
		v = 1
		for move in node.legal_moves:
			cur = copy.copy(node)
			cur.push(move)
			if v == 1:
				v = alphabeta(cur, depth-1, alpha, beta, True) 
			if beta == 1:
				beta = v
			
			v = choiceMove(v, alphabeta(cur, depth-1, alpha, beta, True))[1]
			beta = choiceMove(beta, v)[1] 
			if alpha != -1:
				if choiceMove(alpha, beta)[0] == alpha:
					break
		return v 

# player with white (small letter) pieces
def whiteMove(board, depth):
	alpha = -1
	beta = 1
	v = -1
	for move in board.legal_moves:
		cur = copy.copy(board)
		cur.push(move)
		if v == -1:
			v = alphabeta(cur, depth-1, alpha, beta, False)
			bestMove = move
			if alpha == -1:
				alpha = v
		else:
			new_v = choiceMove(alphabeta(cur, depth-1, alpha, beta, False), v)[0]
			if new_v != v:
				bestMove = move
				v = new_v
			alpha = choiceMove(alpha, v)[0] 

	print(bestMove)	
	board.push(bestMove)
	return board
    
# player with black (capital letter) pieces
def blackMove(board):
	while True:
		try:
			move = raw_input("Enter your move \n")
			board.push_san(move)
			break
		except ValueError:
			print("Illegal move, please try again")

	return board

# main function for the game interface
def playGame():
	moveTotal = 0;
	board = chess.Board()
	depth = raw_input("Enter search depth \n")
	depth = int(depth)
	while board.is_game_over() == False:
		print(board)
		if moveTotal % 2 == 1:
			board = blackMove(board)
		else:
			board =	whiteMove(board, depth)
		moveTotal = moveTotal+1
	
	print(board)
print("Game is over")
