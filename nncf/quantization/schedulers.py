"""
 Copyright (c) 2019 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import logging

from nncf.algo_selector import Registry
from nncf.compression_method_api import CompressionScheduler, CompressionLevel

logger = logging.getLogger(__name__)

QUANTIZATION_SCHEDULERS = Registry("quantization_schedulers")


@QUANTIZATION_SCHEDULERS.register("staged")
class StagedQuantizationScheduler(CompressionScheduler):
    def __init__(self, quantization_ctrl: 'QuantizationController', params=None):
        super().__init__()
        if params is None:
            params = {}
        self.algo = quantization_ctrl
        self.activations_quant_start_epoch = params.get('activations_quant_start_epoch', 1)
        self.weights_quant_start_epoch = params.get('weights_quant_start_epoch', 1)
        self._set_quantization_status()

    @property
    def current_epoch(self):
        # Should be last_epoch + 1, because epoch_step is called in the end of epoch.
        # If last_epoch = -1 and start_epoch = 0, it means that quantizers should be enabled from the very beginning.
        return self.last_epoch + 1

    def epoch_step(self, epoch=None):
        super().epoch_step(epoch)
        should_call_init = False
        if self.current_epoch == self.activations_quant_start_epoch:
            logger.info('Enabled quantization of activations')
            self.algo.enable_activation_quantization()
            should_call_init = True

        if self.current_epoch == self.weights_quant_start_epoch:
            logger.info('Enabled quantization of weights')
            self.algo.enable_weight_quantization()
            should_call_init = True

        if should_call_init:
            self.algo.init_range()

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        # Just enables/disables quantizers without calling initialization of ranges, because it's called on epoch_step
        # in the end of previous epoch before saving the scheduler's state dict.
        self._set_quantization_status()

    def _set_quantization_status(self):
        if self.current_epoch >= self.activations_quant_start_epoch:
            self.algo.enable_activation_quantization()
            logger.info('Enabled quantization of activations')
        else:
            self.algo.disable_activation_quantization()
            logger.info('Disabled quantization of activations')
        if self.current_epoch >= self.weights_quant_start_epoch:
            self.algo.enable_weight_quantization()
            logger.info('Enabled quantization of weights')
        else:
            self.algo.disable_weight_quantization()
            logger.info('Disabled quantization of weights')

    def _calc_density_level(self):
        raise NotImplementedError

    def compression_level(self):
        is_activations_enabled = self.current_epoch >= self.activations_quant_start_epoch
        is_weights_enabled = self.current_epoch >= self.weights_quant_start_epoch
        if is_activations_enabled and is_weights_enabled:
            return CompressionLevel.FULL
        if not is_activations_enabled and not is_weights_enabled:
            return CompressionLevel.NONE
        return CompressionLevel.PARTIAL
