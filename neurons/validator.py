# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


import time
import torch
import json
import os
import deepspeed
import logging
logging.basicConfig(level=logging.INFO)

# Bittensor
import bittensor as bt

# Bittensor Validator Template:
import transverse
from transverse.validator import forward, sync_model, load_dataset
from transverse.multimodal.model.agent import DeepSpeedAgent
from transverse.multimodal.model.anyToImageVideoAudio import TransVerseModel
from transverse.multimodal.model.embedding import EmbeddingModel
from transformers.deepspeed import HfDeepSpeedConfig
from transverse.multimodal.config import load_config

# import base validator class which takes care of most of the boilerplate
from transverse.base.validator import BaseValidatorNeuron


class Validator(BaseValidatorNeuron):
    """
    Your validator neuron class. You should use this class to define your validator's behavior. In particular, you should replace the forward function with your own logic.

    This class inherits from the BaseValidatorNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a validator such as keeping a moving average of the scores of the miners and using them to set weights at the end of each epoch. Additionally, the scores are reset for new hotkeys at the end of each epoch.
    """

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        bt.logging.info("load_state()")
        self.load_state()

        #TODO: Load configs
        self.config.model_config = load_config(self.config)
        self.config.ds_config_path = f'transverse/multimodal/dsconfig/stage_{self.config.stage}.json'
        self.config.ds_config = json.load(open(self.config.ds_config_path))

        # Initialize distributed env
        self.config['master_ip'] = os.getenv('MASTER_ADDR', 'localhost')
        self.config['master_port'] = os.getenv('MASTER_PORT', '6000')
        self.config['world_size'] = int(os.getenv('WORLD_SIZE', '1'))
        self.config['local_rank'] = int(os.getenv('RANK', '0')) % torch.cuda.device_count()
        device = self.config['local_rank'] % torch.cuda.device_count()
        torch.cuda.set_device(device)
        deepspeed.init_distributed(dist_backend='nccl')

        # Load the dataset
        # TODO: Need to download batches from HF
        bt.logging.info("### Loading the dataset ###")
        load_dataset(self) # Arg list: batch_size, dschf_conf, train_micro_batch_size_per_gpu, dataset_name_list
        self.train_iter_operator = iter(self.train_iter)

        # TODO(developer): Anything specific to your use case you can do here
        # TODO: Load the transverse model 
        # TODO: Sync with trained model (Can be done in version 2)
        self.model = TransVerseModel(**self.config)
        delta_ckpt = torch.load('ckpt/pretrained_ckpt/7b_tiva_v0/pytorch_model.pt', map_location=torch.device('cpu'))
        self.model.load_state_dict(delta_ckpt, strict=False)
        bt.logging.info('Loaded the pretrained checkpoint successfully')
        self.agent = DeepSpeedAgent(self.model, self.config)
        # self.embed_model = EmbeddingModel(self.config)

    async def forward(self):
        """
        Validator forward pass. Consists of:
        - Generating the query
        - Querying the miners
        - Getting the responses
        - Rewarding the miners
        - Updating the scores
        """
        # TODO(developer): Rewrite this function based on your protocol definition.
        return await forward(self)


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with Validator() as validator:
        bt.logging.info("Running validator on subnet %d"%validator.config.netuid)
        last_block = 0
        while True:
            if validator.block % 5 == 0 and validator.block > last_block:
                log = (
                    f"Block: {validator.block} | " +
                    "Stake:%.02f | "%(validator.metagraph.S[validator.uid]) +
                    "Rank:%.03f | "%validator.metagraph.R[validator.uid] +
                    "Trust:%.03f | "%validator.metagraph.Tv[validator.uid] +
                    "Consensus:%.03f | "%validator.metagraph.C[validator.uid] +
                    "Incentive:%.03f | "%validator.metagraph.I[validator.uid] +
                    "Emission:%.03f"%validator.metagraph.E[validator.uid]
                )
                bt.logging.info(log)
                last_block = validator.block

            # bt.logging.info("Validator running...", time.time())
            time.sleep(5)
