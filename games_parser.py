import chess
import chess.pgn
import random
import pickle
import argparse
import sys
import numpy as np
import itertools as it
import gzip

def read_games(path):
    pgn = open(path, encoding="utf-8-sig", errors="surrogateescape")
    while True:
        try:
            game = chess.pgn.read_game(pgn)
        except:
            break
        yield game

def complete_game_states( board, generator):
    
    '''
    Stores every state of a board in a list.
    Every move changes the state of the board.
    '''
    
    states_list = [] 
    
    for moves in range(5):
        ''' 
        The first five moves are ignored.
        '''
        try :
            board.push(next(generator))
        except:
            break
            
    while(True):
        '''
        Untill "Stop Iteration" error is thrown
        '''
        try :
            move = next(generator)
            if  bool(board.is_capture(move)) == False:
                states_list.append(board.copy())
            board.push(move)
        except:
            break
            
    return states_list

def complete_game_states( board, generator):
    
    '''
    Stores every state of a board in a list.
    Every move changes the state of the board.
    '''
    
    states_list = [] 
    
    for moves in range(5):
        ''' 
        The first five moves are ignored.
        '''
        try :
            board.push(next(generator))
        except:
            break
            
    while(True):
        '''
        Untill "Stop Iteration" error is thrown
        '''
        try :
            move = next(generator)
            if  bool(board.is_capture(move)) == False:
                states_list.append(board.copy())
            board.push(move)
        except:
            break
            
    return states_list


def bit_string_gen(a):
    bit_string = list( map(lambda x: 1 if x in a else 0, range(0, 64)) )
    return bit_string

def bit_board_gen(given_position, baseboard, winner):
    bit_string = []
    for color in [True,False]:
        for piece in range(1,7):
            
            indices = list(baseboard.pieces(piece, color))
            bit_string.extend(bit_string_gen(indices))

    # Checking the side to move. 1 for white and 0 for black
    bit_string.extend(list(map(int, [given_position.turn])))

    # Checking castling rights of white and balck
    bit_string.extend(list( map( int,
                                      [ 
        given_position.has_kingside_castling_rights(True),
        given_position.has_queenside_castling_rights(True),
        given_position.has_kingside_castling_rights(False),
        given_position.has_queenside_castling_rights(False)  
            ] ) ) )
    bit_string.extend(winner)
    return bit_string

# def initialise_array():
    # position = "2kr3r/pp1n2p1/2p2q1p/2b1p3/4B1b1/2PP1N2/PP3PPP/R1BQ1RK1" 
    # position_full = "r3k2r/pp1n2pp/2p2q2/2b1p3/4B1b1/3P1N2/PPP2PPP/R1BQ1RK1 b kq - 3 11"
    # baseboard = chess.BaseBoard(position)            
    # return bit_board_gen(chess.Board(position_full), baseboard)

def bit_board_array(game, winner):

    bit_list = []
    board = chess.Board() 
    move_generator = game.main_line()
    states_list = complete_game_states(board, move_generator)

    k = min(len(states_list),10)

    '''Taking 10 random positions from the game'''
    random_states = random.sample(states_list, k)

    for state in random_states:

        # FEN represntation of the board
        position = state.board_fen() 

        # A board representing the position of chess pieces.
        baseboard = chess.BaseBoard(position) 

        bit_list.append( bit_board_gen( state, baseboard, list(winner) ) )

    return bit_list

#######################################################################    

def main(engine_name):
    
    games_generator = read_games("./{0}.pgn".format(engine_name))
    bit_list = []
 
    while(True):
        try:
            game = next(games_generator)
            result =  game.headers['Result']
            if result == "1-0":
                bit_list.extend(bit_board_array(game, winner = "1") ) 
            elif result == "0-1":
                bit_list.extend(bit_board_array(game, winner = "0" ) )
            else:
                continue
        except:
            break

    bit_list = np.asarray(bit_list, dtype=np.int32)

    with gzip.GzipFile("{0}.npy.gz".format(engine_name), "w") as f:
        np.save(file=f , arr=bit_list)
#######################################################################
        
main(str(sys.argv[1]))

#######################################################################

# To load back in
# with gzip.GzipFile('Morphy.npy.gz', "r") as f:
    # item = np.load(f)
# Counting white and black wins
# print(item.shape)
# y_ = item[:, -1:].reshape([-1])
# print(np.unique(y_ , return_counts=True))

