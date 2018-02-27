import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from logging import getLogger
from multiprocessing import Manager
from time import sleep
from threading import Lock
from collections import deque

from chess_zero.agent.model_chess import ChessModel
from chess_zero.agent.player_chess import ChessPlayer
from chess_zero.config import Config, INIT_STATE
import chess_zero.env.chess_env as env
from chess_zero.lib.data_helper import get_next_generation_model_dirs
from chess_zero.lib.model_helper import save_as_best_model, load_best_model_weight

logger = getLogger(__name__)
job_done = Lock()
thr_free = Lock()
ng_score = None
cut_white = None
futures =[]

def start(config: Config):
    return EvaluateWorker(config).start()

class EvaluateWorker:
    def __init__(self, config: Config):
        """
        :param config:
        """
        self.config = config
        self.play_config = config.eval.play_config
        self.current_model = self.load_current_model()
        self.ng_model = ChessModel(self.config)
        self.m = Manager()
        self.cur_pipes = self.m.list([self.current_model.get_pipe() for _ in range(self.play_config.max_processes)])
        self.ng_pipes = self.m.list([self.ng_model.get_pipe() for _ in range(self.play_config.max_processes)])
        self.model_list = []
        self.history_list = []

    def start(self):
        while True:
            model_dir,config_path, weight_path = self.load_next_generation_model()
            logger.debug("start evaluate model %s" % (model_dir))
            ng_is_great = self.evaluate_model()
            if ng_is_great:
                logger.debug("New Model become best model: %s" % (model_dir))
                save_as_best_model(self.ng_model)
                self.current_model.load(config_path, weight_path)
            else:
                logger.debug("No need to renew the Model.")
            # self.move_model(model_dir)

    def evaluate_model(self):
        global futures
        global job_done
        global thr_free
        global ng_score
        global cut_white

        job_done.acquire(True)

        game_idx = 0
        futures = []

        with ProcessPoolExecutor(max_workers=self.play_config.max_processes) as executor:
            for k in range(self.play_config.max_processes):
                fut = executor.submit(play_game, self.config, cur=self.cur_pipes, ng=self.ng_pipes, current_white=(game_idx % 2 == 0))
                game_idx += 1
                fut.add_done_callback(recall_fn)
                futures.append(fut)

            results = []
            while True:
                job_done.acquire(True)


                results.append(ng_score)
                win_rate = sum(results) / len(results)

                if (cut_white):
                    player = 'black'
                else:
                    player = 'red'

                logger.debug("game %3d: ng_score=%.1f as %s "
                             "%5.2f\n" % (k, ng_score, player, win_rate))


                if len(results)-sum(results) >= self.config.eval.game_num * (1-self.config.eval.replace_rate):
                    logger.debug("lose count reach %d so give up challenge" % (results.count(0)))
                elif sum(results) >= self.config.eval.game_num * self.config.eval.replace_rate:
                    logger.debug("win count reach %d so change best model" % (results.count(1)))
                elif game_idx < self.config.eval.game_num:
                    fut = executor.submit(play_game, self.config, cur=self.cur_pipes, ng=self.ng_pipes,
                                          current_white=(game_idx % 2 == 0))
                    game_idx += 1
                    fut.add_done_callback(recall_fn)
                    futures.append(fut)

                thr_free.release()
                if len(futures) == 0:
                    break

        win_rate = sum(results) / len(results)
        logger.debug("winning rate %.1f" % win_rate*100)
        return win_rate >= self.config.eval.replace_rate

    def move_model(self, model_dir):
        rc = self.config.resource
        new_dir = os.path.join(rc.next_generation_model_dir, "copies", model_dir)
        os.rename(model_dir, new_dir)

    def load_current_model(self):
        model = ChessModel(self.config)
        load_best_model_weight(model)
        return model

    def load_next_generation_model(self):
        rc = self.config.resource
        while True:
            dirs = get_next_generation_model_dirs(self.config.resource)

            i = -1
            if dirs is not None:
                i = len(dirs)-1
            while i >= 0:
                if dirs[i] in self.history_list:
                    break
                i = i-1
            if (dirs is not None) and (len(dirs) > i+1):
                self.model_list.extend(dirs[i+1:])
                self.history_list.extend(dirs[i+1:])
            if len(self.model_list) > 0:
                break
            logger.info("There is no next generation model to evaluate, waiting for 600s")
            sleep(600)
        model_dir = self.model_list.pop()

        print ('========================================================')
        print ('eval against %s' % (model_dir))
        print ('========================================================')

        config_path = os.path.join(model_dir, rc.next_generation_model_config_filename)
        weight_path = os.path.join(model_dir, rc.next_generation_model_weight_filename)
        self.ng_model.load(config_path, weight_path)
        return model_dir, config_path, weight_path

def recall_fn(future):
    global thr_free
    global job_done
    global ng_score
    global cut_white
    global futures

    thr_free.acquire(True)
    ng_score, cut_white = future.result()
    futures.remove(future)
    job_done.release()


def play_game(config, cur, ng, current_white: bool) -> (float, bool):
    cur_pipe = cur.pop()
    ng_pipe = ng.pop()

    current_player = ChessPlayer(config, pipe=cur_pipe, play_config=config.eval.play_config)
    ng_player = ChessPlayer(config, pipe=ng_pipe, play_config=config.eval.play_config)
    if current_white:
        white, black = current_player, ng_player
    else:
        white, black = ng_player, current_player

    state = INIT_STATE
    steps = 0
    v = 0

    while v == 0:
        if (steps%2) == 0:
            action, policy = white.action(state, steps)
        else:
            action, policy = black.action(state, steps)
        state = env.step(state, action)
        steps += 1
        if steps >= config.eval.max_game_length:
            v = env.testeval(state)
        else:
            v = env.game_over(state)

    white.close()
    black.close()

    if (steps%2) > 0:
        v = -v
    if current_white:
        v = -v

    if v > 0.001:
        ng_score = 1
    elif v < -0.001:
        ng_score = 0
    else:
        ng_score = 0.5
    cur.append(cur_pipe)
    ng.append(ng_pipe)
    return ng_score, current_white
