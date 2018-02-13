import os
import numpy as np


class PlayWithHumanConfig:
    def __init__(self):
        self.simulation_num_per_move = 1200
        self.threads_multiplier = 2
        self.c_puct = 1 # lower  = prefer mean action value
        self.noise_eps = 0
        self.tau_decay_rate = 0  # start deterministic mode
        self.resign_threshold = None

    def update_play_config(self, pc):
        """
        :param PlayConfig pc:
        :return:
        """
        pc.simulation_num_per_move = self.simulation_num_per_move
        pc.search_threads *= self.threads_multiplier
        pc.c_puct = self.c_puct
        pc.noise_eps = self.noise_eps
        pc.tau_decay_rate = self.tau_decay_rate
        pc.resign_threshold = self.resign_threshold
        pc.max_game_length = 999999


class Options:
    new = False


class ResourceConfig:
    def __init__(self):
        self.project_dir = os.environ.get("PROJECT_DIR", _project_dir())
        self.data_dir = os.environ.get("DATA_DIR", _data_dir())

        self.model_dir = os.environ.get("MODEL_DIR", os.path.join(self.data_dir, "model"))
        self.model_best_config_path = os.path.join(self.model_dir, "model_best_config.json")
        self.model_best_weight_path = os.path.join(self.model_dir, "model_best_weight.h5")

        self.model_best_distributed_ftp_server = "alpha-chess-zero.mygamesonline.org"
        self.model_best_distributed_ftp_user = "2537576_chess"
        self.model_best_distributed_ftp_password = "alpha-chess-zero-2"
        self.model_best_distributed_ftp_remote_path = "/alpha-chess-zero.mygamesonline.org/"

        self.next_generation_model_dir = os.path.join(self.model_dir, "next_generation")
        self.next_generation_model_dirname_tmpl = "model_%s"
        self.next_generation_model_config_filename = "model_config.json"
        self.next_generation_model_weight_filename = "model_weight.h5"

        self.play_data_dir = os.path.join(self.data_dir, "play_data")
        self.play_data_filename_tmpl = "play_%s.json"

        self.log_dir = os.path.join(self.project_dir, "logs")
        self.main_log_path = os.path.join(self.log_dir, "main.log")

    def create_directories(self):
        dirs = [self.project_dir, self.data_dir, self.model_dir, self.play_data_dir, self.log_dir,
                self.next_generation_model_dir]
        for d in dirs:
            if not os.path.exists(d):
                os.makedirs(d)


def flipped_uci_labels():
    def repl(x):
        return "".join([(str(9 - int(a)) if a.isdigit() else a) for a in x])

    return [repl(x) for x in create_uci_labels()]


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


class Config:
    labels = create_uci_labels()
    n_labels = int(len(labels))
    flipped_labels = flipped_uci_labels()
    unflipped_index = None

    def __init__(self, config_type="mini"):
        self.opts = Options()
        self.resource = ResourceConfig()

        if config_type == "mini":
            import chess_zero.configs.mini as c
        elif config_type == "normal":
            import chess_zero.configs.normal as c
        elif config_type == "distributed":
            import chess_zero.configs.distributed as c
        else:
            raise RuntimeError('unknown config_type: %s' % (config_type))
        self.model = c.ModelConfig()
        self.play = c.PlayConfig()
        self.play_data = c.PlayDataConfig()
        self.trainer = c.TrainerConfig()
        self.eval = c.EvaluateConfig()
        self.labels = Config.labels
        self.n_labels = Config.n_labels
        self.flipped_labels = Config.flipped_labels

    @staticmethod
    def flip_policy(pol):
        return np.asarray([pol[ind] for ind in Config.unflipped_index])


Config.unflipped_index = [Config.labels.index(x) for x in Config.flipped_labels]


# print(Config.labels)
# print(Config.flipped_labels)


def _project_dir():
    d = os.path.dirname
    return d(d(d(os.path.abspath(__file__))))


def _data_dir():
    return os.path.join(_project_dir(), "data")

RED, BLACK = 0, 1
BORDER, SPACE = 15, 56
LOCAL, OTHER = 0, 1
NETWORK, AI = 0, 1
KING, ADVISOR, BISHOP, KNIGHT, ROOK, CANNON, PAWN, NONE = 0, 1, 2, 3, 4, 5, 6, -1

AI_SEARCH_DEPTH = 5

BOARD_HEIGHT = 10
BOARD_WIDTH = 9



# init_fen = '1R2k4/4a3r/b1n5b/6p1p/p3PP2c/2r4C1/P5R1P/N8/6N2/2BAKA3 r - - 0 1'
# init_fen = '1n7/5k3/5a2b/9/2brp4/1pp5p/9/B2A5/4K4/4r4 r - - 0 1'
# init_fen = '3aka3/9/C7n/2p4r1/2n6/P3p2pP/2P3P2/R2RK3B/9/3A1A3 r - - 0 1'
# init_fen = 'rn2ka1nr/4a4/bc2C4/2p1p1p1p/p2c5/2B6/P1P1P1P1P/1C7/9/RN1AKABNR r - - 0 1'
# init_fen = 'rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR r - - 0 1'
INIT_STATE = 'rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR'
mov_dir = {
    'k': [(0, -1), (1, 0), (0, 1), (-1, 0)],
    'K': [(0, -1), (1, 0), (0, 1), (-1, 0)],
    'a': [(-1, -1), (1, -1), (-1, 1), (1, 1)],
    'A': [(-1, -1), (1, -1), (-1, 1), (1, 1)],
    'b': [(-2, -2), (2, -2), (2, 2), (-2, 2)],
    'B': [(-2, -2), (2, -2), (2, 2), (-2, 2)],
    'n': [(-1, -2), (1, -2), (2, -1), (2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1)],
    'N': [(-1, -2), (1, -2), (2, -1), (2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1)],
    'P': [(0, -1), (-1, 0), (1, 0)],
    'p': [(0, 1), (-1, 0), (1, 0)]}

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
    move_str += chr(ord('a')+ x)
    move_str += str(y)
    move_str += chr(ord('a')+ x_)
    move_str += str(y_)
    return move_str

def str_to_move(act:str):
    arr = [0]*4
    b = ['a','0','a','0']
    for i in range(4):
        arr[i] = ord(act[i])-ord(b[i])
    return arr