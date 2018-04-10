import enum
from chess_zero.config import *
import numpy as np
import copy

from logging import getLogger

logger = getLogger(__name__)

# noinspection PyArgumentList
Winner = enum.Enum("Winner", "black white draw")

# input planes
# noinspection SpellCheckingInspection
pieces_order = 'KABNRCPkabnrcp' # 14x10x9

ind = {pieces_order[i]: i for i in range(14)}

def game_over(state):
    if 'k' not in state:
        return int(-1)
    if 'K' not in state:
        return int(1)
    return int(0)

def testeval(state) -> float:
    piece_vals = {'r': 14, 'n': 7, 'b': 3, 'a': 2, 'k':1, 'c': 5, 'p': 1} # for RED account
    ans = 0.0
    tot = 0
    for c in state:
        if not c.isalpha():
            continue

        if c.islower():
            ans += piece_vals[c]
            tot += piece_vals[c]
        else:
            ans -= piece_vals[c.lower()]
            tot += piece_vals[c.lower()]
    v = ans/tot
    return np.tanh(v * 3) # arbitrary


def canon_input_planes(state):
    history_both = to_planes(state)

    ret = history_both
    assert ret.shape == (14, BOARD_HEIGHT, BOARD_WIDTH)
    return ret

def flip_state(state):
    rows = state.split('/')
    def swapcase(a):
        if a.isalpha():
            return a.lower() if a.isupper() else a.upper()
        return a
    def swapall(aa):
        return "".join([swapcase(a) for a in aa])

    return "/".join([swapall(row) for row in reversed(rows)])


def to_planes(state):
    san = fen_to_san(state)
    pieces_both = np.zeros(shape=(14, BOARD_HEIGHT, BOARD_WIDTH), dtype=np.float32)
    for rank in range(BOARD_HEIGHT):
        for file in range(BOARD_WIDTH):
            v = san[rank * BOARD_WIDTH + file]
            if v.isalpha():
                pieces_both[ind[v]][rank][file] = 1
    assert pieces_both.shape == (14, BOARD_HEIGHT, BOARD_WIDTH)
    return pieces_both


def fen_to_san(fen):
    san = fen
    san = san.replace("2", "11")
    san = san.replace("3", "111")
    san = san.replace("4", "1111")
    san = san.replace("5", "11111")
    san = san.replace("6", "111111")
    san = san.replace("7", "1111111")
    san = san.replace("8", "11111111")
    san = san.replace("9", "111111111")
    return san.replace("/", "")

def san_to_fen(san):
    fen = ''
    k = 0
    z = 0
    for i in range(BOARD_HEIGHT):
        for j in range(BOARD_WIDTH):
            ch = san[k]
            k += 1
            if ch.isdigit():
                z = z+int(ch)
            else:
                if (z > 0):
                    fen = fen+str(z)
                    z = 0
                fen = fen+ch
        if (z > 0):
            fen = fen + str(z)
            z = 0
        if i < BOARD_HEIGHT-1:
            fen = fen+'/'

    return fen


def step(state: str, action: str):
    mov_arr = str_to_move(action)
    u = mov_arr[1]*BOARD_WIDTH + mov_arr[0]
    v = mov_arr[3]*BOARD_WIDTH + mov_arr[2]
    san = fen_to_san(state)
    san = san[:v]+san[u]+san[v+1:]
    san = san[:u]+'1'+san[u+1:]
    fen = san_to_fen(san)
    return flip_state(fen)


def fen_to_board(fen):
    board = [['.' for col in range(BOARD_WIDTH)] for row in range(BOARD_HEIGHT)]
    x = 0
    y = 0
    for ch in fen:

        if ch == ' ':
            break
        if ch == '/':
            x = 0
            y += 1
        elif ch >= '1' and ch <= '9':
            for i in range(int(ch)):
                board[y][x] = '.'
                x = x + 1
        else:
            board[y][x] = ch
            x = x + 1
    return board


def legal_moves(fen):
    board = fen_to_board(fen)
    legal_moves = []
    for y in range(BOARD_HEIGHT):
        for x in range(BOARD_WIDTH):
            ch = board[y][x]
            if not ch.islower():
                continue
            if ch in mov_dir:
                for d in mov_dir[ch]:
                    x_ = x + d[0]
                    y_ = y + d[1]
                    if not can_move(board, x_, y_):
                        continue
                    elif ch == 'p' and y < 5 and x_ != x:  # for red pawn
                        continue
                    elif ch == 'n' or ch == 'b' : # for knight and bishop
                        if board[y+int(d[1]/2)][x+int(d[0]/2)] != '.':
                            continue
                        elif ch == 'b' and y_ > 4:
                            continue
                    elif ch == 'k' or ch == 'a': # for king and advisor
                        if x_ < 3 or x_ > 5:
                            continue
                        if y_ > 2:
                            continue
                    legal_moves.append(move_to_str(x, y, x_, y_))
                    if (ch == 'k'): #for King to King check
                        d, u = y_board_from(board, x, y)
                        if (u < BOARD_HEIGHT and board[u][x] == 'K'):
                            legal_moves.append(move_to_str(x, y, x, u))

            elif ch == 'r' or ch == 'c': # for connon and root
                l,r = x_board_from(board,x,y)
                d,u = y_board_from(board,x,y)
                for x_ in range(l+1,x):
                    legal_moves.append(move_to_str(x, y, x_, y))
                for x_ in range(x+1,r):
                    legal_moves.append(move_to_str(x, y, x_, y))
                for y_ in range(d+1,y):
                    legal_moves.append(move_to_str(x, y, x, y_))
                for y_ in range(y+1,u):
                    legal_moves.append(move_to_str(x, y, x, y_))
                if ch == 'r': # for root
                    if can_move(board, l, y):
                        legal_moves.append(move_to_str(x, y, l, y))
                    if can_move(board, r, y):
                        legal_moves.append(move_to_str(x, y, r, y))
                    if can_move(board, x, d):
                        legal_moves.append(move_to_str(x, y, x, d))
                    if can_move(board, x, u):
                        legal_moves.append(move_to_str(x, y, x, u))
                else: # for connon
                    l_, _ = x_board_from(board, l,y)
                    _, r_ = x_board_from(board, r,y)
                    d_, _ = y_board_from(board, x,d)
                    _, u_ = y_board_from(board, x,u)
                    if can_move(board, l_, y):
                        legal_moves.append(move_to_str(x, y, l_, y))
                    if can_move(board, r_, y):
                        legal_moves.append(move_to_str(x, y, r_, y))
                    if can_move(board, x, d_):
                        legal_moves.append(move_to_str(x, y, x, d_))
                    if can_move(board, x, u_):
                        legal_moves.append(move_to_str(x, y, x, u_))
    return legal_moves

def can_move(board,x,y): # basically check the move
    if x < 0 or x > BOARD_WIDTH-1:
        return False
    if y < 0 or y > BOARD_HEIGHT-1:
        return False
    if board[y][x].islower():
        return False
    return True

def x_board_from(board,x,y):
    l = x-1
    r = x+1
    while l > -1 and board[y][l] == '.':
        l = l-1
    while r < BOARD_WIDTH and board[y][r] == '.':
        r = r+1
    return l,r

def y_board_from(board,x,y):
    d = y-1
    u = y+1
    while d > -1 and board[d][x] == '.':
        d = d-1
    while u < BOARD_HEIGHT and board[u][x] == '.':
        u = u+1
    return d,u
