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
import numpy as np

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from utility_functions import weighted_binary_crossentropy, sensitivity, specificity
from tf2_net import GenNet


class SimpleTrainer(Executor):
    def __init__(self, epochs_per_round):
        super().__init__()
        self.epochs_per_round = epochs_per_round
        self.train_images, self.train_labels = None, None
        self.test_images, self.test_labels = None, None
        self.model = None

        self.datapath = os.getcwd() + "/"
        self.inputsize = 100


    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.setup(fl_ctx)

    def setup(self, fl_ctx: FLContext):
        simupath = '/home/avanhilten/PycharmProjects/nvidia_conference/NVFlare/examples/hello-GenNet_tf2/custom/'
        client_name = fl_ctx.get_identity_name()
        if client_name == "site-1":
            self.xtrain = np.load(simupath + 'Simulations/xtrain_1.npy')
            self.ytrain = np.load(simupath + 'Simulations/ytrain_1.npy')
        elif client_name == "site-2":
            self.xtrain = np.load(simupath + 'Simulations/xtrain_2.npy')
            self.ytrain = np.load(simupath + 'Simulations/ytrain_2.npy')
        elif client_name == "site-3":
            self.xtrain = np.load(simupath + 'Simulations/xtrain_3.npy')
            self.ytrain = np.load(simupath + 'Simulations/ytrain_3.npy')

        self.xval = np.load(simupath + 'Simulations/xval.npy')
        self.yval = np.load(simupath + 'Simulations/yval.npy')

        model = GenNet()
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


        # update local model weights with received weights

        self.model.set_weights(list(model_weights.values()))

        # adjust LR or other training time info as needed
        # such as callback in the fit function
        self.model.fit(x=self.xtrain, y=self.ytrain, batch_size=64, epochs=self.epochs_per_round, verbose=1,
                  validation_data=(self.xval, self.yval), shuffle=True)

        # report updated weights in shareable
        weights = {self.model.get_layer(index=key).name: value for key, value in enumerate(self.model.get_weights())}
        dxo = DXO(data_kind=DataKind.WEIGHTS, data=weights)

        self.log_info(fl_ctx, "Local epochs finished. Returning shareable")
        new_shareable = dxo.to_shareable()
        return new_shareable

def weighted_binary_crossentropy(y_true, y_pred):
    weight_positive_class = 1.2
    weight_negative_class = 1

    y_true = tf.keras.backend.clip(y_true, 0.0001, 1)
    y_pred = tf.keras.backend.clip(y_pred, 0.0001, 1)

    return tf.keras.backend.mean(
        -y_true * tf.keras.backend.log(y_pred + 0.0001) * weight_positive_class - (1 - y_true) * tf.keras.backend.log(
            1 - y_pred + 0.0001) * weight_negative_class)
