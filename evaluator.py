import chess
import chess.pgn
import math 
import network
import tensorflow as tf
import games_parser as gp
import itertools as it
import numpy as np
import tfdeploy as td
import time

mode = td.Model("model.pkl")
x1, x2, y = mode.get("position_one", "position_two", "answer")

def eval_func(pos1, pos2): 
    r = y.eval({x1:pos1, x2:pos2})
    return r
    
def get_bit( board ):
    '''
    Input :
        board = copy of the board on which the game is played
        move = "e2e4" style 
    '''
    # FEN represntation of the board
    position = board.board_fen() 
    # A board representing the position of chess pieces.
    baseboard = chess.BaseBoard(position) 
    bit_b = gp.bit_board_gen(board, baseboard, "0")
    bit_b.pop()
    return np.array([bit_b])
    
def better( move1, move2 ):
    index = np.argmax( eval_func(get_bit(move1), get_bit(move2) )[0] ) 
    return (move1, move2)[index]

def worse( move1, move2 ):
    index = np.argmin( eval_func(get_bit(move1), get_bit(move2) )[0] ) 
    return (move1, move2)[index]
    
def argmaxi(state, function):
    
    def pusher(board, move):
        b = board.copy()
        b.push(move)
        return b

    seq = list( map( lambda move: pusher(state, move) , state.legal_moves  ) ) 
    
    best_move, best_score = seq[0], function(seq[0])

    for node in (seq):
        value = function(node)
        if better(best_move, node) == node :
            best_move, best_score = node, value
            
    return best_move
    
def alphabeta_search(state, d=15, cutoff_test=None):
    """Search game to determine best action; use alpha-beta pruning.
    This version cuts off search and uses an evaluation function."""

    def max_value(state, alpha, beta, depth):
        if cutoff_test(state, depth):
            return state
        v = -math.inf
        for mv in state.legal_moves:
            s = state.copy()
            s.push(mv)
            
            if v == -math.inf:
                v = min_value(s,alpha,beta,depth+1)
            if beta ==  math.inf:   
                beta = v
                
            v = better(v, min_value(s, alpha, beta, depth+1))
            if beta != math.inf:
                if better(v, beta) == v :
                    return v
            alpha = better(alpha, v)
        return v

    def min_value(state, alpha, beta, depth):
        if cutoff_test(state, depth):
            return state
        v = math.inf
        for mv in state.legal_moves:
            s = state.copy()
            s.push(mv)
            
            if v == math.inf:
                v = max_value(s,alpha,beta,depth+1)
            if alpha == -math.inf:
                alpha = v
                
            v = worse(v, max_value(s, alpha, beta, depth+1))
            if alpha != -math.inf:
                if worse(v, alpha) == v :
                    return v
            beta = worse(beta, v)
        return v
   
    cutoff_test = (cutoff_test or
                   (lambda state,depth: depth>d or state.is_game_over()))
    
    action  = argmaxi(state, lambda board : min_value(board, -math.inf, math.inf, 0))
    return action
    
def users_move(board):
    while True:
        try:
            move = input("Enter your move in ''Nb1a3'' style \n")
            board.push_san(move)
            break
        except ValueError:
            print("Ambiguous Move..")
    return board
    
def play():
    moveTotal = 0;
    board = chess.Board()
    while board.is_game_over() == False:
        if moveTotal % 2 == 1:
            board = users_move(board)
        else:
            st = time.time()
            board = anthony_joshua(board)
            print("MoveTime:", time.time() -st)
        print(display(board))
        moveTotal = moveTotal+1
    print("Game-over")
    
print(anthony_joshua(chess.Board('rqb2rk1/3nbppp/p2pp3/6P1/1p1BPP2/2NB1Q2/PPP4P/2KR3R w - - 0 16'))