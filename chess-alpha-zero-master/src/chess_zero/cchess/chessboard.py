#! /usr/bin/env python
# -*- coding: utf-8 -*-

# pycchess - just another chinese chess UI
# Copyright (C) 2011 - 2015 timebug

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from chess_zero.cchess.common import *


class Chessboard:

    def __init__(self, board=None):
        if (board is None):
            self.height = 10
            self.width = 9
            self.board = [['.' for col in range(self.width)] for row in range(self.height)]
            self.assign_fen(None)
        else:
            self.turn = board.turn
            self.history = board.history.copy()
            self.ate = board.ate.copy()
            self.height = board.height
            self.width = board.width
            self.board = board.board.copy()

    def _resign(self):
        self.turn = RED
        self.history = []
        self.ate = []
        self._legal_moves = None
        self._fen = None

    def _update(self):
        self._fen = None
        self._legal_moves = None
        if len(self.history)%2 == 0:
            self.turn = RED
        else:
            self.turn = BLACK

    def fen(self):
        if self._fen is not None:
            return self._fen

        c = 0
        fen = ''
        for i in range(self.height):
            c = 0
            for j in range(self.width):
                if self.board[i][j] == '.':
                    c = c+1
                else:
                    if (c > 0):
                        fen = fen+str(c)
                    fen = fen+self.board[i][j]
                    c = 0
            if (c > 0):
                fen = fen + str(c)
            if (i < self.height-1):
                fen = fen+'/'
        if self.turn is RED:
            fen += ' r'
        else:
            fen += ' b'
        fen += ' - - 0 1'
        self._fen = fen
        return self._fen

    @property
    def legal_moves(self):
        if self._legal_moves is not None:
            return self._legal_moves

        legal_moves = []
        for y in range(self.height):
            for x in range(self.width):
                ch = self.board[y][x]
                if (self.turn == RED and ch.isupper()):
                    continue
                if (self.turn == BLACK and ch.islower()):
                    continue
                if ch in mov_dir:
                    if (x == 0 and y == 3):
                        aa = 3
                    for d in mov_dir[ch]:
                        x_ = x + d[0]
                        y_ = y + d[1]
                        if not self._can_move(x_, y_):
                            continue
                        elif ch == 'p' and y < 5 and x_ != x:  # for red pawn
                            continue
                        elif ch == 'P' and y > 4 and x_ != x:  # for black pawn
                            continue
                        elif ch == 'n' or ch == 'N' or ch == 'b' or ch == 'B': # for knight and bishop
                            if self.board[y+int(d[1]/2)][x+int(d[0]/2)] != '.':
                                continue
                            elif ch == 'b' and y_ > 4:
                                continue
                            elif ch == 'B' and y_ < 5:
                                continue
                        elif ch != 'p' and ch != 'P': # for king and advisor
                            if x_ < 3 or x_ > 5:
                                continue
                            if (ch == 'k' or ch == 'a') and y_ > 2:
                                continue
                            if (ch == 'K' or ch == 'A') and y_ < 7:
                                continue
                        legal_moves.append(move_to_str(x, y, x_, y_))
                        if (ch == 'k' and self.turn == RED): #for King to King check
                            d, u = self._y_board_from(x, y)
                            if (self.board[u][x] == 'K'):
                                legal_moves.append(move_to_str(x, y, x, u))
                        elif (ch == 'K' and self.turn == BLACK):
                            d, u = self._y_board_from(x, y)
                            if (self.board[d][x] == 'k'):
                                legal_moves.append(move_to_str(x, y, x, d))
                elif ch != '.': # for connon and root
                    l,r = self._x_board_from(x,y)
                    d,u = self._y_board_from(x,y)
                    for x_ in range(l+1,x):
                        legal_moves.append(move_to_str(x, y, x_, y))
                    for x_ in range(x+1,r):
                        legal_moves.append(move_to_str(x, y, x_, y))
                    for y_ in range(d+1,y):
                        legal_moves.append(move_to_str(x, y, x, y_))
                    for y_ in range(y+1,u):
                        legal_moves.append(move_to_str(x, y, x, y_))
                    if ch == 'r' or ch == 'R': # for root
                        if self._can_move(l, y):
                            legal_moves.append(move_to_str(x, y, l, y))
                        if self._can_move(r, y):
                            legal_moves.append(move_to_str(x, y, r, y))
                        if self._can_move(x, d):
                            legal_moves.append(move_to_str(x, y, x, d))
                        if self._can_move(x, u):
                            legal_moves.append(move_to_str(x, y, x, u))
                    else: # for connon
                        l_, _ = self._x_board_from(l,y)
                        _, r_ = self._x_board_from(r,y)
                        d_, _ = self._y_board_from(x,d)
                        _, u_ = self._y_board_from(x,u)
                        if self._can_move(l_, y):
                            legal_moves.append(move_to_str(x, y, l_, y))
                        if self._can_move(r_, y):
                            legal_moves.append(move_to_str(x, y, r_, y))
                        if self._can_move(x, d_):
                            legal_moves.append(move_to_str(x, y, x, d_))
                        if self._can_move(x, u_):
                            legal_moves.append(move_to_str(x, y, x, u_))

        self._legal_moves = legal_moves
        return self._legal_moves

    def is_legal(self, mov):
        return mov.uci in self.legal_moves

    def push_uci(self, uci):
        mov = Move(uci)
        self.push(mov)

    def push(self, mov):
        if self.is_legal(mov):
            self.ate.append(self.board[mov.n[1]][mov.n[0]])
            self.board[mov.n[1]][mov.n[0]] = self.board[mov.p[1]][mov.p[0]]
            self.board[mov.p[1]][mov.p[0]] = '.'
            self.history.append(mov)
            self._update()

    def pop(self):
        mov = self.history.pop()
        self.board[mov.p[1]][mov.p[0]] = self.board[mov.n[1]][mov.n[0]]
        self.board[mov.n[1]][mov.n[0]] = self.ate.pop()
        self._update()

    def assign_fen(self, fen):
        self._resign()
        if fen is None:
            fen = init_fen
        x = 0
        y = 0
        for k in range(0, len(fen)):
            ch = fen[k]
            if ch == ' ':
                if (fen[k+1] == 'b'):
                    self.turn = BLACK
                break
            if ch == '/':
                x = 0
                y += 1
            elif ch >= '1' and ch <= '9':
                for i in range(int(ch)):
                    self.board[y][x] = '.'
                    x = x+1
            else:
                self.board[y][x] = ch
                x = x+1
        self._fen = fen

    def _is_same_side(self,x,y):
        if self.turn == RED and self.board[y][x].islower():
            return True
        if self.turn == BLACK and self.board[y][x].isupper():
            return True

    def _can_move(self,x,y): # basically check the move
        if x < 0 or x > self.width-1:
            return False
        if y < 0 or y > self.height-1:
            return False
        if self._is_same_side(x,y):
            return False
        return True

    def _x_board_from(self,x,y):
        l = x-1
        r = x+1
        while l > -1 and self.board[y][l] == '.':
            l = l-1
        while r < self.width and self.board[y][r] == '.':
            r = r+1
        return l,r

    def _y_board_from(self,x,y):
        d = y-1
        u = y+1
        while d > -1 and self.board[d][x] == '.':
            d = d-1
        while u < self.height and self.board[u][x] == '.':
            u = u+1
        return d,u

    def result(self, claim_draw=True):
        rst = '*'
        if ('k' not in self.board[0]) and ('k' not in self.board[1]) and ('k' not in self.board[2]):
            rst = '0-1'
        if ('K' not in self.board[9]) and ('K' not in self.board[8]) and ('K' not in self.board[7]):
            rst = '1-0'
        return rst

if __name__ == '__main__': # test
    board = Chessboard()
    print(board.legal_moves)