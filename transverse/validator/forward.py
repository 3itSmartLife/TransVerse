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

import bittensor as bt
import torch
import time

from transverse.protocol import DistributedTraining, UpdateModel
from transverse.validator.reward import get_rewards
from transverse.utils.uids import get_random_uids
# from transverse.validator.dataset import random_batch, extract_embeddings

async def forward(self):
    """
    The forward function is called by the validator every time step.

    It is responsible for querying the network and scoring the responses.

    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.

    """
    # TODO(developer): Define how the validator selects a miner to query, how often, etc.
    # get_random_uids is an example method, but you can replace it with your own.

    # TODO: Check the request mode
    # Implement random selection of two modes
    self.config.request_mode = 'compute'

    # TODO: Define validation mechanism
    if self.config.request_mode == 'compute':
        # TODO: Aggregate the gradients and update the model
        pass

        miner_uids = get_random_uids(self, k=self.config.neuron.sample_size)

        # TODO: Adjust random_batch function call
        # Select training batch randomly and extract embeddings
        # training_batch = random_batch(self.config)
        try:
            train_batch = next(self.train_iter_operator)
        except:
            self.train_iter_operator = iter(self.train_iter)
            train_batch = next(self.train_iter_operator)

        # TODO: Adjust extract_embeddings function call
        # Extract embeddings from the random batch
        # embeddings_batch = extract_embeddings(training_batch)
        # Mock training batch to the miner
        # embeddings_batch = [bt.Tensor.serialize(torch.ones(77, 768, dtype=torch.float).to('cpu'))]
        train_emb_batch = train_batch.pop('caption_embs')
        train_emb_batch = [bt.Tensor.serialize(emb) for emb in train_emb_batch]
        bt.logging.info(f'Training batch: {train_batch}')

        bt.logging.info(f'Sending challenges to miners: {miner_uids}')
        # The dendrite client queries the network.
        responses = await self.dendrite(
            # Send the query to selected miner axons in the network.
            axons=[self.metagraph.axons[uid] for uid in miner_uids],
            synapse=DistributedTraining(train_batch=train_batch, train_emb_batch=train_emb_batch),
            # All responses have the deserialize function called on them before returning.
            # You are encouraged to define your own deserialization function.
            deserialize=True,
            timeout=self.config.neuron.timeout
        )

        # Log the results for monitoring purposes.
        bt.logging.info(f"Received gradients from miners")
        gradients = []
        for response in responses:
            if response:
                gradients.append([bt.Tensor.deserialize(grad) for grad in response])

        if len(gradients) > 0:
            # TODO: Average gradients
            bt.logging.info('Averaging gradients ...')
            accum_grads = []
            for i in range(len(gradients[0])):
                accum_grads.append([grad[i] for grad in gradients])
            avg_grads = [torch.stack(grads).mean(dim=0) for grads in accum_grads]
            try:
                print(avg_grads[0].shape)
            except:
                print(avg_grads[0].shape())

            torch.save(avg_grads, 'grads.dump')
            avg_grads = [bt.Tensor.serialize(grad) for grad in avg_grads]
            # TODO: Propagate averaged gradients to miners
            bt.logging.info(f'Sending gradients to miners to update the model')
            responses = await self.dendrite(
                # Send the query to selected miner axons in the network.
                axons=[self.metagraph.axons[uid] for uid in miner_uids],
                # Construct a dummy query. This simply contains a single integer.
                synapse=UpdateModel(avg_gradients=avg_grads),
                # All responses have the deserialize function called on them before returning.
                # You are encouraged to define your own deserialization function.
                deserialize=True,
                timeout=self.config.neuron.timeout
            )

            if True in responses:
                bt.logging.info('Miner models successfully updated')
            else:
                bt.logging.info('Miner models update failed')


    elif self.config.requst_mode == 'validate':
        # TODO: Calculate gradients on the validator side
        pass

    # TODO(developer): Define how the validator scores responses.
    # Adjust the scores based on responses from miners.
    # rewards = get_rewards(self, query=self.step, responses=responses).to(self.device)

    # bt.logging.info(f"Scored responses: {rewards}")
    # Update the scores based on the rewards. You may want to define your own update_scores function for custom behavior.
    # miner_uids = miner_uids.to(self.device)
    # self.update_scores(rewards, miner_uids)