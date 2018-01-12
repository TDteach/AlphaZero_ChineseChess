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

from common import *
from chessboard import *
from chessnet import *
import json
import os
import time
import random

import pygame
#import pygame._view
from pygame.locals import *

import sys
from subprocess import PIPE, Popen
from threading import Thread
from Queue import Queue, Empty

ON_POSIX = 'posix' in sys.builtin_module_names

def enqueue_output(out, queue):
    for line in iter(out.readline, ''):
        queue.put(line)
    out.close()

pygame.init()

screen = pygame.display.set_mode(size, 0, 32)
chessboard = chessboard()

replay=None
all_replay = None
ind_td = None
step = 0

def random_select_replay():
    global all_replay
    global replay
    global ind_td
    k = random.choice([i for i in range(len(ind_td)-1)])
    replay = all_replay[ind_td[k]:ind_td[k+1]]

if len(sys.argv) == 2 and sys.argv[1][:2] == '-r':
    global all_replay
    global ind_td
    replay_dir = '/home/tdteach/workspace/AlphaZero_ChineseChess/chess-alpha-zero-master/data/play_data'
    files = os.listdir(replay_dir)
    files.sort()
    file_path = replay_dir+'/'+files[-1]
    try:
        with open(file_path, "rt") as f:
            data= json.load(f)
    except Exception as e:
        print(e)
    all_replay = [d[0] for d in data]
    ind_td=[]
    for i in range(len(all_replay)):
        if all_replay[i] == all_replay[0]:
            ind_td.append(i)
    ind_td.append(len(all_replay))
    random_select_replay()
    #p = Popen("./harmless", stdin=PIPE, stdout=PIPE, close_fds=ON_POSIX)
    #(chessboard.fin, chessboard.fout) = (p.stdin, p.stdout)


elif len(sys.argv) == 2 and sys.argv[1][:2] == '-n':
    chessboard.net = chessnet()

    if sys.argv[1][2:] == 'r':
        pygame.display.set_caption("red")
        chessboard.side = RED
    elif sys.argv[1][2:] == 'b':
        pygame.display.set_caption("black")
        chessboard.side = BLACK
    else:
        print '>> quit game'
        sys.exit()

    chessboard.net.NET_HOST = sys.argv[2]

elif len(sys.argv) == 1:
    p = Popen("./harmless", stdin=PIPE, stdout=PIPE, close_fds=ON_POSIX)
    (chessboard.fin, chessboard.fout) = (p.stdin, p.stdout)
    q = Queue()
    t = Thread(target=enqueue_output, args=(chessboard.fout, q))
    t.daemon = True
    t.start()

    chessboard.fin.write("ucci\n")
    chessboard.fin.flush()

    while True:
        try:
            output = q.get_nowait()
        except Empty:
            continue
        else:
            sys.stdout.write(output)
            if 'ucciok' in output:
                break

    chessboard.mode = AI
    pygame.display.set_caption("harmless")
    chessboard.side = RED
else:
    print '>> quit game'
    sys.exit()

chessboard.fen_parse(fen_str)
init = True
waiting = False
moved = False

def fertilize(compact):
    compact = compact.replace('9', '.........')
    compact = compact.replace('8', '........')
    compact = compact.replace('7', '.......')
    compact = compact.replace('6', '......')
    compact = compact.replace('5', '.....')
    compact = compact.replace('4', '....')
    compact = compact.replace('3', '...')
    compact = compact.replace('2', '..')
    compact = compact.replace('1', '.')
    compact = compact.replace('/', '')
    return compact

def newGame():
    global init
    global waiting
    global moved
    global step

    step = 0

    print "setoption newgame\n"
    print '>> new game'

    chessboard.fen_parse(fen_str)
    init = True
    waiting = False
    moved = False

def quitGame():
    if chessboard.mode is NETWORK:
        net = chessnet()
        net.send_move('quit')
    if chessboard.mode is AI:
        chessboard.fin.write("quit\n")
        chessboard.fin.flush()
        p.terminate()

    print '>> quit game'
    sys.exit()

def runGame():
    global init
    global waiting
    global moved
    global replay

    for event in pygame.event.get():
        if event.type == QUIT:
            quitGame()
        if event.type == KEYDOWN:
            if event.key == K_SPACE:
                if len(sys.argv) == 2 and sys.argv[1][:2] == '-r':
                    if not waiting or chessboard.over:
                        newGame()
                        random_select_replay()
                    return
                if chessboard.mode == AI:
                    if not waiting or chessboard.over:
                        newGame()
                        return

        if event.type == MOUSEBUTTONDOWN:
            x, y = pygame.mouse.get_pos()
            if x < BORDER or x > (WIDTH - BORDER):
                break
            if y < BORDER or y > (HEIGHT - BORDER):
                break
            x = (x - BORDER) / SPACE
            y = (y - BORDER) / SPACE
            print (x,y)
            if not waiting and not chessboard.over:
                moved = chessboard.move_chessman(x, y)
                if chessboard.mode == NETWORK and moved:
                    chessboard.over = chessboard.game_over(1-chessboard.side)
                    if chessboard.over:
                        chessboard.over_side = 1-chessboard.side

    chessboard.draw(screen)
    pygame.display.update()
    # time.sleep(3)

    if moved:
        if chessboard.mode is NETWORK:
            move_str = chessboard.net.get_move()
            if move_str is not 'quit':
                # print 'recv move: %s' % move_str
                move_arr = str_to_move(move_str)
            else:
                quitGame()

        if chessboard.mode is AI:
            try:
                output = q.get_nowait()
            except Empty:
                waiting = True
                return
            else:
                waiting = False
                sys.stdout.write(output)

            if output[0:10] == 'nobestmove':
                chessboard.over = True
                chessboard.over_side = 1 - chessboard.side

                if chessboard.over_side == RED:
                    win_side = 'BLACK'
                else:
                    win_side = 'RED'
                print '>>', win_side, 'win'

                return
            elif output[0:8] == 'bestmove':
                move_str = output[9:13]
                move_arr = str_to_move(move_str)
            else:
                return

        chessboard.side = 1 - chessboard.side
        chessboard.move_from = OTHER
        chessboard.move_chessman(move_arr[0], move_arr[1])
        chessboard.move_chessman(move_arr[2], move_arr[3])
        chessboard.move_from = LOCAL
        chessboard.side = 1 - chessboard.side

        # if chessboard.check(chessboard.side):
        chessboard.over = chessboard.game_over(chessboard.side)
        if chessboard.over:
            chessboard.over_side = chessboard.side

            if chessboard.over_side == RED:
                win_side = 'BLACK'
            else:
                win_side = 'RED'
            print '>>', win_side, 'win'

        moved = False

    if len(sys.argv) == 2 and sys.argv[1][:2] == '-n' and init:
        move_str = chessboard.net.get_move()
        if move_str is not None:
            # print 'recv move: %s' % move_str
            move_arr = str_to_move(move_str)

            chessboard.side = 1 - chessboard.side
            chessboard.move_from = OTHER
            chessboard.move_chessman(move_arr[0], move_arr[1])
            chessboard.move_chessman(move_arr[2], move_arr[3])
            chessboard.move_from = LOCAL
            chessboard.side = 1 - chessboard.side
            init = False
        else:
            chessboard.over = True

    if len(sys.argv) == 2 and sys.argv[1][:2] == '-r': # for replay
        bef = fertilize(replay[0])
        aft = fertilize(replay[1])
        arr=[0,0,0,0]
        k=0
        for y in range(10):
            for x in range(9):
                if bef[k] != aft[k]:
                    if (aft[k] == '.'):
                        arr[0] = x
                        arr[1] = 9-y
                    else:
                        arr[2] = x
                        arr[3] = 9-y
                k = k+1
        print 'from ('+str(arr[0])+','+str(9-arr[1])+') to ('+str(arr[2])+','+str(9-arr[3])+')'
        chessboard.move_chessman(arr[0], arr[1])
        chessboard.move_chessman(arr[2], arr[3])
        chessboard.side = 1 - chessboard.side
        chessboard.move_from = 1-chessboard.move_from
        if len(replay) > 2:
            replay = replay[1:]
        else:
            #raw_input('next game type enter:')
            newGame()
            random_select_replay()

try:
    step = 0
    while True:
        if step%2 == 0:
            print 'RED'
        else:
            print 'BLACK'
        step = step+1
        print 'STEP '+str(step)+':'
        runGame()
        #raw_input('pause')
        #time.sleep(1)
except KeyboardInterrupt:
    quitGame()
