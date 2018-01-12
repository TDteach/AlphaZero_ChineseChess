import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from logging import getLogger
from multiprocessing import Manager
from time import sleep
from collections import deque

from chess_zero.agent.model_chess import ChessModel
from chess_zero.agent.player_chess import ChessPlayer
from chess_zero.config import Config
from chess_zero.env.chess_env import ChessEnv, Winner
from chess_zero.lib.data_helper import get_next_generation_model_dirs
from chess_zero.lib.model_helper import save_as_best_model, load_best_model_weight
import time

logger = getLogger(__name__)


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
        self.cur_pipes = self.m.list([self.current_model.get_pipes(self.play_config.search_threads) for _ in range(self.play_config.max_processes)])
        self.ng_pipes = self.m.list([self.ng_model.get_pipes(self.play_config.search_threads) for _ in range(self.play_config.max_processes)])
        self.model_list = []

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

        futures = deque()
        with ProcessPoolExecutor(max_workers=self.play_config.max_processes) as executor:
            for game_idx in range(self.config.eval.game_num):
                fut = executor.submit(play_game, self.config, cur=self.cur_pipes, ng=self.ng_pipes, current_white=(game_idx % 2 == 0))
                futures.append(fut)

            results = []
            for game_idx in range(self.config.eval.game_num):
                # ng_score := if ng_model win -> 1, lose -> 0, draw -> 0.5
                fut = futures.popleft()
                ng_score, env, current_white = fut.result()
                results.append(ng_score)
                win_rate = sum(results) / len(results)
                game_idx = len(results)

                if (current_white):
                    player = 'red'
                else:
                    player = 'black'
                if (env.resigned):
                    resigned = 'by resign '
                else:
                    resigned = '          '

                logger.debug("game %3d: ng_score=%.1f as %s "
                             "%s"
                             "%5.1f\n"
                             "%s" % (game_idx, ng_score, player, resigned, win_rate, env.board.fen().split(' ')[0]))

                # colors = ("current_model", "ng_model")
                # if not current_white:
                #     colors = reversed(colors)
                # pretty_print(env, colors)

                if len(results)-sum(results) >= self.config.eval.game_num * (1-self.config.eval.replace_rate):
                    logger.debug("lose count reach %d so give up challenge" % (results.count(0)))
                    return False
                if sum(results) >= self.config.eval.game_num * self.config.eval.replace_rate:
                    logger.debug("win count reach %d so change best model" % (results.count(1)))
                    return True

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
                if dirs[i] in self.model_list:
                    break
                i = i-1
            if (dirs is not None) and (len(dirs) > i):
                self.model_list.extend(dirs[i+1:])
            if len(self.model_list) > 0:
                break
            logger.info("There is no next generation model to evaluate, waiting for 600s")
            sleep(600)
        model_dir = self.model_list.pop()

        config_path = os.path.join(model_dir, rc.next_generation_model_config_filename)
        weight_path = os.path.join(model_dir, rc.next_generation_model_weight_filename)
        self.ng_model.load(config_path, weight_path)
        return model_dir, config_path, weight_path


def play_game(config, cur, ng, current_white: bool) -> (float, ChessEnv, bool):
    cur_pipes = cur.pop()
    ng_pipes = ng.pop()
    env = ChessEnv().reset()

    current_player = ChessPlayer(config, pipes=cur_pipes, play_config=config.eval.play_config)
    ng_player = ChessPlayer(config, pipes=ng_pipes, play_config=config.eval.play_config)
    if current_white:
        white, black = current_player, ng_player
    else:
        white, black = ng_player, current_player

    while not env.done:
        if env.white_to_move:
            action = white.action(env)
        else:
            action = black.action(env)
        env.step(action)
        if env.num_halfmoves >= config.eval.max_game_length:
            env.adjudicate()

    if env.winner == Winner.draw:
        ng_score = 0.5
    elif env.white_won == current_white:
        ng_score = 0
    else:
        ng_score = 1
    cur.append(cur_pipes)
    ng.append(ng_pipes)
    return ng_score, env, current_white
