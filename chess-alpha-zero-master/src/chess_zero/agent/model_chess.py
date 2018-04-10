import ftplib
import hashlib
import json
import os
from logging import getLogger


import tensorflow as tf
from keras import backend as KK

config = tf.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=8, allow_soft_placement=True, device_count = {'CPU':8})
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list='0'
session = tf.Session(config=config)
KK.set_session(session)

from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

from chess_zero.agent.api_chess import ChessModelAPI
from chess_zero.config import Config
from filelock import FileLock

# noinspection PyPep8Naming

logger = getLogger(__name__)


class ChessModel:
    def __init__(self, config: Config):
        self.config = config
        self.model = None  # type: Model
        self.digest = None
        self.api = None

    def get_pipes(self, num = 1):
        if self.api is None:
            self.api = ChessModelAPI(self.config, self)
            self.api.start()
        return [self.api.get_pipe() for _ in range(num)]

    def get_pipe(self):
        if self.api is None:
            self.api = ChessModelAPI(self.config, self)
            self.api.start()
        return self.api.get_pipe()

    def build(self):
        mc = self.config.model
        # in_x = x = Input((18, 8, 8))
        in_x = x = Input((14,10,9)) # change to CC

        # (batch, channels, height, width)
        x = Conv2D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_first_filter_size, padding="same",
                   data_format="channels_first", use_bias=False, kernel_regularizer=l2(mc.l2_reg),
                   name="input_conv-"+str(mc.cnn_first_filter_size)+"-"+str(mc.cnn_filter_num))(x)
        x = BatchNormalization(axis=1, name="input_batchnorm")(x)
        x = Activation("relu", name="input_relu")(x)

        for i in range(mc.res_layer_num):
            x = self._build_residual_block(x, i + 1)

        res_out = x
        
        # for policy output
        x = Conv2D(filters=2, kernel_size=1, data_format="channels_first", use_bias=False, kernel_regularizer=l2(mc.l2_reg),
                    name="policy_conv-1-2")(res_out)
        x = BatchNormalization(axis=1, name="policy_batchnorm")(x)
        x = Activation("relu", name="policy_relu")(x)
        x = Flatten(name="policy_flatten")(x)
        # no output for 'pass'
        policy_out = Dense(self.config.n_labels, kernel_regularizer=l2(mc.l2_reg), activation="softmax", name="policy_out")(x)

        # for value output
        x = Conv2D(filters=4, kernel_size=1, data_format="channels_first", use_bias=False, kernel_regularizer=l2(mc.l2_reg),
                    name="value_conv-1-4")(res_out)
        x = BatchNormalization(axis=1, name="value_batchnorm")(x)
        x = Activation("relu",name="value_relu")(x)
        x = Flatten(name="value_flatten")(x)
        x = Dense(mc.value_fc_size, kernel_regularizer=l2(mc.l2_reg), activation="relu", name="value_dense")(x)
        value_out = Dense(1, kernel_regularizer=l2(mc.l2_reg), activation="tanh", name="value_out")(x)

        self.model = Model(in_x, [policy_out, value_out], name="chess_model")

    def _build_residual_block(self, x, index):
        mc = self.config.model
        in_x = x
        res_name = "res"+str(index)
        x = Conv2D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_filter_size, padding="same",
                   data_format="channels_first", use_bias=False, kernel_regularizer=l2(mc.l2_reg), 
                   name=res_name+"_conv1-"+str(mc.cnn_filter_size)+"-"+str(mc.cnn_filter_num))(x)
        x = BatchNormalization(axis=1, name=res_name+"_batchnorm1")(x)
        x = Activation("relu",name=res_name+"_relu1")(x)
        x = Conv2D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_filter_size, padding="same",
                   data_format="channels_first", use_bias=False, kernel_regularizer=l2(mc.l2_reg), 
                   name=res_name+"_conv2-"+str(mc.cnn_filter_size)+"-"+str(mc.cnn_filter_num))(x)
        x = BatchNormalization(axis=1, name="res"+str(index)+"_batchnorm2")(x)
        x = Add(name=res_name+"_add")([in_x, x])
        x = Activation("relu", name=res_name+"_relu2")(x)
        return x

    @staticmethod
    def fetch_digest(weight_path):
        if os.path.exists(weight_path):
            m = hashlib.sha256()
            with open(weight_path, "rb") as f:
                m.update(f.read())
            return m.hexdigest()

    def load(self, config_path, weight_path):
        mc = self.config.model
        resources = self.config.resource
        if mc.distributed and config_path == resources.model_best_config_path:
            try:
                logger.debug("loading model from server")
                ftp_connection = ftplib.FTP(resources.model_best_distributed_ftp_server,
                                            resources.model_best_distributed_ftp_user,
                                            resources.model_best_distributed_ftp_password)
                ftp_connection.cwd(resources.model_best_distributed_ftp_remote_path)
                ftp_connection.retrbinary("RETR model_best_config.json", open(config_path, 'wb').write)
                ftp_connection.retrbinary("RETR model_best_weight.h5", open(weight_path, 'wb').write)
                ftp_connection.quit()
            except:
                pass
        if os.path.exists(config_path) and os.path.exists(weight_path):
            logger.debug("loading model from %s" % (config_path))
            with FileLock(config_path+'.lock'):
                with open(config_path, "rt") as f:
                    self.model = Model.from_config(json.load(f))
            with FileLock(weight_path+'.lock'):
                self.model.load_weights(weight_path)
                self.model._make_predict_function()
                self.digest = self.fetch_digest(weight_path)
                logger.debug("loaded model digest = %s" % (self.digest))
            return True
        else:
            logger.debug("model files does not exist at %s and %s" % (config_path, weight_path))
            return False

    def save(self, config_path, weight_path):
        logger.debug("saving model to %s" % (config_path))
        print('debug-3')
        with FileLock(config_path+'.lock'):
            with open(config_path, "wt") as f:
                print('debug-2')
                json.dump(self.model.get_config(), f)
                print('debug-1')
        with FileLock(weight_path+'.lock'):
            self.model.save_weights(weight_path)
            print('debug-0')
            self.digest = self.fetch_digest(weight_path)
            logger.debug("saved model digest %s" % (self.digest))

        print('debug')

        mc = self.config.model
        resources = self.config.resource
        print('debug2')
        if mc.distributed and config_path == resources.model_best_config_path:
            try:
                print('debug3')
                logger.debug("saving model to server")
                ftp_connection = ftplib.FTP(resources.model_best_distributed_ftp_server,
                                            resources.model_best_distributed_ftp_user,
                                            resources.model_best_distributed_ftp_password)
                ftp_connection.cwd(resources.model_best_distributed_ftp_remote_path)
                fh = open(config_path, 'rb')
                ftp_connection.storbinary('STOR model_best_config.json', fh)
                fh.close()

                fh = open(weight_path, 'rb')
                ftp_connection.storbinary('STOR model_best_weight.h5', fh)
                fh.close()
                ftp_connection.quit()
            except:
                print('debug4')
                pass
