import enum
# import chess.pgn
from chess_zero.cchess.chessboard import Chessboard
from chess_zero.cchess.common import *
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


class ChessEnv:

    def __init__(self):
        self.board = None
        self.num_halfmoves = 0
        self.winner = None  # type: Winner
        self.resigned = False
        self.result = None

    def reset(self):
        # self.board = chess.Board()
        self.board = Chessboard()
        self.num_halfmoves = 0
        self.winner = None
        self.resigned = False
        return self

    def update(self, board):
        self.board = Chessboard(board)
        self.winner = None
        self.resigned = False
        return self

    @property
    def done(self):
        return self.winner is not None

    @property
    def white_won(self):
        return self.winner == Winner.white

    @property
    def white_to_move(self):
        return self.board.turn == RED

    def step(self, action: str, check_over = True):
        """
        :param action:
        :param check_over:
        :return:
        """
        if check_over and action is None:
            self._resign()
            return

        self.board.push_uci(action)

        self.num_halfmoves += 1

        if check_over and self.board.result(claim_draw=True) != "*":
            self._game_over()

    def _game_over(self):
        if self.winner is None:
            self.result = self.board.result(claim_draw=True)
            if self.result == '1-0':
                self.winner = Winner.white
            elif self.result == '0-1':
                self.winner = Winner.black
            else:
                self.winner = Winner.draw

    def _resign(self):
        self.resigned = True
        if self.white_to_move: # WHITE RESIGNED!
            self.winner = Winner.black
            self.result = "0-1"
        else:
            self.winner = Winner.white
            self.result = "1-0"

    def adjudicate(self):
        score = self.testeval(absolute = True)
        if abs(score) < 0.01:
            self.winner = Winner.draw
            self.result = "1/2-1/2"
        elif score > 0:
            self.winner = Winner.white
            self.result = "1-0"
        else:
            self.winner = Winner.black
            self.result = "0-1"

    def ending_average_game(self):
        self.winner = Winner.draw
        self.result = "1/2-1/2"

    def copy(self):
        env = copy.deepcopy(self)
        env.board = copy.deepcopy(self.board)
        return env

    def render(self):
        print("\n")
        print(self.board)
        print("\n")

    @property
    def observation(self):
        return self.board.fen()

    # def deltamove(self, fen_next):
    #     moves = list(self.board.legal_moves)
    #     for mov in moves:
    #         self.board.push(mov)
    #         fee = self.board.fen()
    #         self.board.pop()
    #         if fee == fen_next:
    #             return mov.uci()
    #     return None

    def replace_tags(self):
        return replace_tags_board(self.board.fen())

    def canonical_input_planes(self):
        return canon_input_planes(self.board.fen())

    def testeval(self, absolute=False) -> float:
        return testeval(self.board.fen(), absolute)

def testeval(fen, absolute = False) -> float:
    piece_vals = {'r': 14, 'n': 7, 'b': 3, 'a': 2, 'k':1, 'c': 5, 'p': 1} # for RED account
    ans = 0.0
    tot = 0
    for c in fen.split(' ')[0]:
        if not c.isalpha():
            continue

        if c.islower():
            ans += piece_vals[c]
            tot += piece_vals[c]
        else:
            ans -= piece_vals[c.lower()]
            tot += piece_vals[c.lower()]
    v = ans/tot
    if not absolute and is_black_turn(fen):
        v = -v
    assert abs(v) < 1
    return np.tanh(v * 3) # arbitrary


def check_current_planes(realfen, planes):
    cur = planes[0:12]
    assert cur.shape == (12, 8, 8)
    fakefen = ["1"] * 64
    for i in range(12):
        for rank in range(8):
            for file in range(8):
                if cur[i][rank][file] == 1:
                    assert fakefen[rank * 8 + file] == '1'
                    fakefen[rank * 8 + file] = pieces_order[i]

    castling = planes[12:16]
    fiftymove = planes[16][0][0]
    ep = planes[17]

    castlingstring = ""
    for i in range(4):
        if castling[i][0][0] == 1:
            castlingstring += castling_order[i]

    if len(castlingstring) == 0:
        castlingstring = '-'

    epstr = "-"
    for rank in range(8):
        for file in range(8):
            if ep[rank][file] == 1:
                epstr = coord_to_alg((rank, file))

    realfen = maybe_flip_fen(realfen)
    realparts = realfen.split(' ')
    assert realparts[1] == 'w'
    assert realparts[2] == castlingstring
    assert realparts[3] == epstr
    assert int(realparts[4]) == fiftymove
    # realparts[5] is the fifty-move clock, discard that
    return "".join(fakefen) == replace_tags_board(realfen)


def canon_input_planes(fen):
    fen = maybe_flip_fen(fen)
    return all_input_planes(fen)


def all_input_planes(fen):
    # current_aux_planes = aux_planes(fen)

    history_both = to_planes(fen)

    # ret = np.vstack((history_both, current_aux_planes))
    ret = history_both
    assert ret.shape == (14, 10, 9)
    return ret


def maybe_flip_fen(fen):
    if not is_black_turn(fen):
        return fen
    foo = fen.split(' ')
    rows = foo[0].split('/')
    def swapcase(a):
        if a.isalpha():
            return a.lower() if a.isupper() else a.upper()
        return a
    def swapall(aa):
        return "".join([swapcase(a) for a in aa])

    # before: "".join(sorted(swapall(foo[2]))) instead of foo[2]
    return "/".join([swapall(row) for row in reversed(rows)]) \
        + " " + ('r' if foo[1] == 'b' else 'b') \
        + " " + foo[2] \
        + " " + foo[3] + " " + foo[4] + " " + foo[5]


def flip_move(mov:str) -> str:
    return mov[0]+str(ord('9')-ord(mov[1]))+mov[2]+str(ord('9')-ord(mov[3]))


def maybe_flip_moves(moves, flip=False):
    if not flip:
        return moves
    rst = []
    for mov in moves:
        rst.append(flip_move(mov))
    return rst

# def aux_planes(fen):
#     foo = fen.split(' ')
#
#     en_passant = np.zeros((8, 8), dtype=np.float32)
#     if foo[3] != '-':
#         eps = alg_to_coord(foo[3])
#         en_passant[eps[0]][eps[1]] = 1
#
#     fifty_move_count = int(foo[4])
#     fifty_move = np.full((8, 8), fifty_move_count, dtype=np.float32)
#
#     castling = foo[2]
#     auxiliary_planes = [np.full((8, 8), int('K' in castling), dtype=np.float32),
#                         np.full((8, 8), int('Q' in castling), dtype=np.float32),
#                         np.full((8, 8), int('k' in castling), dtype=np.float32),
#                         np.full((8, 8), int('q' in castling), dtype=np.float32),
#                         fifty_move,
#                         en_passant]
#
#     ret = np.asarray(auxiliary_planes, dtype=np.float32)
#     assert ret.shape == (6, 8, 8)
#     return ret

# FEN board is like this:
# a8 b8 .. h8
# a7 b7 .. h7
# .. .. .. ..
# a1 b1 .. h1
# 
# FEN string is like this:
#  0  1 ..  7
#  8  9 .. 15
# .. .. .. ..
# 56 57 .. 63

# my planes are like this:
# 00 01 .. 07
# 10 11 .. 17
# .. .. .. ..
# 70 71 .. 77
#


def alg_to_coord(alg):
    rank = 9 - int(alg[1])        # 0-9
    file = ord(alg[0]) - ord('a') # a-i
    return rank, file


def coord_to_alg(coord):
    letter = chr(ord('a') + coord[1])
    number = str(9 - coord[0])
    return letter + number


def to_planes(fen):
    board_state = replace_tags_board(fen)
    pieces_both = np.zeros(shape=(14, 10, 9), dtype=np.float32)
    for rank in range(10):
        for file in range(9):
            v = board_state[rank * 9 + file]
            if v.isalpha():
                pieces_both[ind[v]][rank][file] = 1
    assert pieces_both.shape == (14, 10, 9)
    return pieces_both


def replace_tags_board(board_san):
    board_san = board_san.split(" ")[0]
    board_san = board_san.replace("2", "11")
    board_san = board_san.replace("3", "111")
    board_san = board_san.replace("4", "1111")
    board_san = board_san.replace("5", "11111")
    board_san = board_san.replace("6", "111111")
    board_san = board_san.replace("7", "1111111")
    board_san = board_san.replace("8", "11111111")
    board_san = board_san.replace("9", "111111111")
    return board_san.replace("/", "")


def is_black_turn(fen):
    return fen.split(" ")[1] == 'b'
