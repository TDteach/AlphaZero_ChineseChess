from multiprocessing import connection, Pipe
from threading import Thread

import numpy as np

from chess_zero.config import Config


class ChessModelAPI:
    # noinspection PyUnusedLocal
    def __init__(self, config: Config, agent_model):  # ChessModel
        self.agent_model = agent_model
        self.pipes = []

    def start(self):
        prediction_worker = Thread(target=self.predict_batch_worker, name="prediction_worker")
        prediction_worker.daemon = True
        prediction_worker.start()

    def get_pipe(self):
        me, you = Pipe(True)
        self.pipes.append(me)
        return you

    def predict_batch_worker(self):
        while True:
            ready = connection.wait(self.pipes,timeout=0.001)
            if not ready:
                continue
            data, result_pipes, ss = [], [], []
            for pipe in ready:
                try:
                    tmp = pipe.recv()
                except EOFError as e:
                    print('Bug Bug Bug')
                data.extend(tmp)
                ss.append(len(tmp))
                result_pipes.append(pipe)
            t_data = np.asarray(data, dtype=np.float32)
            policy_ary, value_ary = self.agent_model.model.predict_on_batch(t_data)
            tt = list()
            k = 0
            i = 0
            for p,v in zip(policy_ary, value_ary):
                tt.append((p,float(v)))
                k = k+1
                if (k >= ss[i]) :
                    result_pipes[i].send(tt)
                    tt = list()
                    i = i+1
                    k = 0
