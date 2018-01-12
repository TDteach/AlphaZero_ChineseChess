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