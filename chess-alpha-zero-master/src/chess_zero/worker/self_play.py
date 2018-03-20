import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from logging import getLogger
from multiprocessing import Manager
from threading import Thread
from time import time
from collections import defaultdict
from threading import Lock
from time import sleep

from chess_zero.agent.model_chess import ChessModel
from chess_zero.agent.player_chess import ChessPlayer, VisitStats
from chess_zero.config import Config, INIT_STATE
import chess_zero.env.chess_env as env
from chess_zero.lib.data_helper import get_game_data_filenames, write_game_data_to_file
from chess_zero.lib.model_helper import load_best_model_weight, save_as_best_model, \
    need_to_reload_best_model_weight

logger = getLogger(__name__)
job_done = Lock()
thr_free = Lock()
rst = None
data = None
futures =[]


def start(config: Config):
    return SelfPlayWorker(config).start()


# noinspection PyAttributeOutsideInit
class SelfPlayWorker:
    def __init__(self, config: Config):
        """
        :param config:
        """
        self.config = config
        self.current_model = self.load_model()
        self.m = Manager()
        self.cur_pipes = self.m.list([self.current_model.get_pipe() for _ in range(self.config.play.max_processes)])

    def start(self):
        global job_done
        global thr_free
        global rst
        global data
        global futures

        self.buffer = []
        need_to_renew_model = True
        job_done.acquire(True)

        with ProcessPoolExecutor(max_workers=self.config.play.max_processes) as executor:
            game_idx = 0
            while True:
                game_idx += 1
                start_time = time()

                if need_to_renew_model and len(futures) == 0:
                    load_best_model_weight(self.current_model)
                    for i in range(self.config.play.max_processes):
                        ff = executor.submit(self_play_buffer, self.config, cur=self.cur_pipes)
                        ff.add_done_callback(recall_fn)
                        futures.append(ff)
                    need_to_renew_model = False

                job_done.acquire(True)

                print("game %3d time=%5.1fs "
                    "%3d %d " % (game_idx, time() - start_time, rst[0], rst[1]))
                print('game %3d time=%5.1fs ' % (game_idx, time()-start_time))

                self.buffer += data

                if (game_idx % self.config.play_data.nb_game_in_file) == 0:
                    self.flush_buffer()
                    if need_to_reload_best_model_weight(self.current_model):
                        need_to_renew_model = True
                    self.remove_play_data(all=False) # remove old data
                if not need_to_renew_model: # avoid congestion
                    ff = executor.submit(self_play_buffer, self.config, cur=self.cur_pipes)
                    ff.add_done_callback(recall_fn)
                    futures.append(ff) # Keep it going
                thr_free.release()

        if len(data) > 0:
            self.flush_buffer()

    def load_model(self):
        model = ChessModel(self.config)
        if self.config.opts.new or not load_best_model_weight(model):
            model.build()
            save_as_best_model(model)
        return model

    def flush_buffer(self):
        rc = self.config.resource
        game_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        path = os.path.join(rc.play_data_dir, rc.play_data_filename_tmpl % game_id)
        logger.info("save play data to %s" % (path))
        thread = Thread(target=write_game_data_to_file, args=(path, self.buffer))
        thread.start()
        self.buffer = []

    def remove_play_data(self,all=False):
        files = get_game_data_filenames(self.config.resource)
        if (all):
            for path in files:
                os.remove(path)
        else:
            while len(files) > self.config.play_data.max_file_num:
                os.remove(files[0])
                del files[0]

def recall_fn(future):
    global thr_free
    global job_done
    global rst
    global data
    global futures

    thr_free.acquire(True)
    rst, data = future.result()
    futures.remove(future)
    job_done.release()

def self_play_buffer(config, cur) -> (tuple, list):
    pipe = cur.pop() # borrow

    player = ChessPlayer(config, pipe=pipe)

    state = INIT_STATE
    history = [state]
    policys = [] 
    v = 0
    steps = 0
    while v == 0:
        no_act = None
        if state in history[:-1]:
            no_act = []
            for i in range(len(history)-1):
                if history[-i] == state:
                    no_act.append(history[-i+1])
        action, policy = player.action(state, steps, no_act)
        policys.append(policy)
        history.append(action)
        state = env.step(state, action)
        steps += 1
        history.append(state)
        if steps >= config.play.max_game_length:
            v = env.testeval(state)
            break
        else:
            v = env.game_over(state)

    player.close()

    if (steps%2) == 1:
        v = -v
    vv = v
    data = []
    for i in range(steps):
        k = i*2
        data.append([history[k], policys[i], v])
        v = -v

    cur.append(pipe)
    return (steps, vv), data
