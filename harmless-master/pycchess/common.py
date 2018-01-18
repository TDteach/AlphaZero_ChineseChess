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

import pygame

size = (WIDTH, HEIGHT) = (530, 586)

RED, BLACK = 0, 1
BORDER, SPACE, ENTER = 15, 56, 13
LOCAL, OTHER = 0, 1
NETWORK, AI = 0, 1
KING, ADVISOR, BISHOP, KNIGHT, ROOK, CANNON, PAWN, NONE = 0, 1, 2, 3, 4, 5, 6, -1

AI_SEARCH_DEPTH = 5

image_path = 'image/'
board_image = 'cchessboard.png'
select_image = 'select.png'
over_image = 'over.png'
done_image = 'done.png'
chessman_image = ['king.png',
                  'advisor.png',
                  'bishop.png',
                  'knight.png',
                  'rook.png',
                  'cannon.png',
                  'pawn.png']

check_sound = "sounds/CHECK.WAV"
move_sound = 'sounds/MOVE.WAV'
capture_sound = 'sounds/CAPTURE.WAV'

fen_str = 'rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1'

king_dir = [(0, -1), (1, 0), (0, 1), (-1, 0)]
advisor_dir = [(-1, -1), (1, -1), (-1, 1), (1, 1)]
bishop_dir = [(-2, -2), (2, -2), (2, 2), (-2, 2)]
knight_dir = [(-1, -2), (1, -2), (2, -1), (2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1)]
rook_dir = [(0, -1), (1, 0), (0, 1), (-1, 0)]
cannon_dir = [(0, -1), (1, 0), (0, 1), (-1, 0)]
pawn_dir = [[(0, -1), (-1, 0), (1, 0)], [(0, 1), (-1, 0), (1, 0)]]

bishop_check = [(-1, -1), (1, -1), (-1, 1), (1, 1)]
knight_check = [(0, -1), (0, -1), (1, 0), (1, 0), (0, 1), (0, 1), (-1, 0), (-1, 0)]

def get_kind(fen_ch):
    if fen_ch in ['k', 'K']:
        return KING
    elif fen_ch in ['a', 'A']:
        return ADVISOR
    elif fen_ch in ['b', 'B']:
        return BISHOP
    elif fen_ch in ['n', 'N']:
        return KNIGHT
    elif fen_ch in ['r', 'R']:
        return ROOK
    elif fen_ch in ['c', 'C']:
        return CANNON
    elif fen_ch in ['p', 'P']:
        return PAWN
    else:
        return NONE

def get_char(kind, color):
    if kind is KING:
        return ['K', 'k'][color]
    elif kind is ADVISOR:
        return ['A', 'a'][color]
    elif kind is BISHOP:
        return ['B', 'b'][color]
    elif kind is KNIGHT:
        return ['N', 'n'][color]
    elif kind is ROOK:
        return ['R', 'r'][color]
    elif kind is CANNON:
        return ['C', 'c'][color]
    elif kind is PAWN:
        return ['P', 'p'][color]
    else:
        return ''

def move_to_str(x, y, x_, y_):
    move_str = ''
    move_str += chr(ord('a') + x)
    move_str += str(9 - y)
    move_str += chr(ord('a') + x_)
    move_str += str(9 - y_)
    return move_str

def str_to_move(move_str):
    move_arr = [0]*4
    move_arr[0] = ord(move_str[0]) - ord('a')
    move_arr[1] = ord('9') - ord(move_str[1])
    move_arr[2] = ord(move_str[2]) - ord('a')
    move_arr[3] = ord('9') - ord(move_str[3])
    return move_arr

class move:
    def __init__(self, p, n):
        self.p = p
        self.n = n

def load_sound(name):
    try:
        sound = pygame.mixer.Sound(name)
    except pygame.error, message:
        raise SystemExit, message
    return sound

def create_uci_labels():
    labels_array = []
    numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] # row
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'] # col

    for n1 in range(10):
        for l1 in range(9):
            destinations = [(n1, t) for t in range(9)] + \
                           [(t, l1) for t in range(10)] + \
                           [(n1 + a, l1 + b) for (a, b) in
                            [(-2, -1), (-1, -2), (-2, 1), (1, -2), (2, -1), (-1, 2), (2, 1), (1, 2)]]
            for (n2, l2) in destinations:
                if (n1, l1) != (n2, l2) and n2 in range(10) and l2 in range(9):
                    move = letters[l1] + numbers[n1] + letters[l2] + numbers[n2]
                    labels_array.append(move)

    #for red advisor
    labels_array.append('d0e1')
    labels_array.append('f0e1')
    labels_array.append('d2e1')
    labels_array.append('f2e1')
    labels_array.append('e1d0')
    labels_array.append('e1f0')
    labels_array.append('e1d2')
    labels_array.append('e1f2')
    # for black advisor
    labels_array.append('d9e8')
    labels_array.append('f9e8')
    labels_array.append('d7e8')
    labels_array.append('f7e8')
    labels_array.append('e8d9')
    labels_array.append('e8f9')
    labels_array.append('e8d7')
    labels_array.append('e8f7')

    #for red bishop
    labels_array.append('c0a2')
    labels_array.append('c0e2')
    labels_array.append('g0e2')
    labels_array.append('g0i2')
    labels_array.append('c4a2')
    labels_array.append('c4e2')
    labels_array.append('g4e2')
    labels_array.append('g4i2')
    labels_array.append('a2c0')
    labels_array.append('e2c0')
    labels_array.append('e2g0')
    labels_array.append('i2g0')
    labels_array.append('a2c4')
    labels_array.append('e2c4')
    labels_array.append('e2g4')
    labels_array.append('i2g4')
    # for black bishop
    labels_array.append('c9a7')
    labels_array.append('c9e7')
    labels_array.append('g9e7')
    labels_array.append('g9i7')
    labels_array.append('c5a7')
    labels_array.append('c5e7')
    labels_array.append('g5e7')
    labels_array.append('g5i7')
    labels_array.append('a7c9')
    labels_array.append('e7c9')
    labels_array.append('e7g9')
    labels_array.append('i7g9')
    labels_array.append('a7c5')
    labels_array.append('e7c5')
    labels_array.append('e7g5')
    labels_array.append('i7g5')


    return labels_array
