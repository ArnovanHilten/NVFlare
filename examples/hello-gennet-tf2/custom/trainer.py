# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=""
import tensorflow as tf
import pathlib
import numpy as np
import os
import pickle
from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from utility_functions import weighted_binary_crossentropy, sensitivity, specificity
from GenNet_tf2 import GenNet



class SimpleTrainer(Executor):
    def __init__(self, epochs_per_round):
        super().__init__()
        self.epochs_per_round = epochs_per_round
        self.train_images, self.train_labels = None, None
        self.test_images, self.test_labels = None, None
        self.model = None
        self.datapath = os.getcwd() + "/"
        self.path_to_file = None
        self.inputsize = 100

        print('selfdatapath', self.datapath)


    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.setup(fl_ctx)

    def setup(self, fl_ctx: FLContext):
        client_name = fl_ctx.get_identity_name()
        run_number = fl_ctx.get_run_number()

        self.path_to_file = pathlib.Path(__file__).parent.resolve()

        if client_name == "site-1":
            self.xtrain = np.load(self.path_to_file / 'Simulations/xtrain_1.npy')
            self.ytrain = np.load(self.path_to_file / 'Simulations/ytrain_1.npy')
        elif client_name == "site-2":
            self.xtrain = np.load(self.path_to_file / 'Simulations/xtrain_2.npy')
            self.ytrain = np.load(self.path_to_file / 'Simulations/ytrain_2.npy')
        elif client_name == "site-3":
            self.xtrain = np.load(self.path_to_file / 'Simulations/xtrain_3.npy')
            self.ytrain = np.load(self.path_to_file / 'Simulations/ytrain_3.npy')

        self.xval = np.load(self.path_to_file / 'Simulations/xval.npy')
        self.yval = np.load(self.path_to_file / 'Simulations/yval.npy')

        model = GenNet(path_run_folder=self.path_to_file)
        optimizer = tf.keras.optimizers.Adam(lr=0.0006)
        model.compile(loss=weighted_binary_crossentropy, optimizer=optimizer,
                      metrics=["accuracy", sensitivity, specificity])

        _ = model(tf.keras.Input(shape=(self.inputsize,)))
        print(model.summary())
        self.var_list = [model.get_layer(index=index).name for index in range(len(model.get_weights()))]
        self.model = model

    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        """
        This function is an extended function from the super class.
        As a supervised learning based trainer, the train function will run
        evaluate and train engines based on model weights from `shareable`.
        After finishing training, a new `Shareable` object will be submitted
        to server for aggregation.

        Args:
            task_name: dispatched task
            shareable: the `Shareable` object acheived from server.
            fl_ctx: the `FLContext` object achieved from server.
            abort_signal: if triggered, the training will be aborted.

        Returns:
            a new `Shareable` object to be submitted to server for aggregation.
        """

        # retrieve model weights download from server's shareable
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        if task_name != "train":
            return make_reply(ReturnCode.TASK_UNKNOWN)

        dxo = from_shareable(shareable)
        model_weights = dxo.data

        # use previous round's client weights to replace excluded layers from server
        prev_weights = {
            self.model.get_layer(index=key).name: value for key, value in enumerate(self.model.get_weights())
        }

        ordered_model_weights = {key: model_weights.get(key) for key in prev_weights}
        for key in self.var_list:
            value = ordered_model_weights.get(key)
            if np.all(value == 0):
                ordered_model_weights[key] = prev_weights[key]

        # update local model weights with received weights
        self.model.set_weights(list(ordered_model_weights.values()))

        # adjust LR or other training time info as needed
        # such as callback in the fit function
        self.model.fit(x=self.xtrain, y=self.ytrain, batch_size=64, epochs=self.epochs_per_round, verbose=1,
                  validation_data=(self.xval, self.yval), shuffle=True)

        # report updated weights in shareable
        weights = {self.model.get_layer(index=key).name: value for key, value in enumerate(self.model.get_weights())}

        run_number = fl_ctx.get_run_number()

        with open(self.datapath + "/run_" + str(run_number) + '/weights.pickle', "wb") as f:
            pickle.dump(weights, f)

        dxo = DXO(data_kind=DataKind.WEIGHTS, data=weights)

        self.log_info(fl_ctx, "Local epochs finished. Returning shareable")
        new_shareable = dxo.to_shareable()
        return new_shareable

def weighted_binary_crossentropy(y_true, y_pred):
    weight_positive_class = 2
    weight_negative_class = 1

    y_true = tf.keras.backend.clip(y_true, 0.0001, 1)
    y_pred = tf.keras.backend.clip(y_pred, 0.0001, 1)

    return tf.keras.backend.mean(
        -y_true * tf.keras.backend.log(y_pred + 0.0001) * weight_positive_class - (1 - y_true) * tf.keras.backend.log(
            1 - y_pred + 0.0001) * weight_negative_class)
