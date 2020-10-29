'''
Utilities for training a deep Q network. Includes:
- Target network
- Experience replay
- Pytorch devices (?)
- Observation history
'''

import gfootball.env as football_env
from collections import deque
import numpy as np
import torch
import random

class HistoryBuffer():
    def __init__(self, n, obs_shape=(4, 96, 72), fill_value=0., device=None):
        '''
        Set up the history buffer.

        Inputs: 
          - n: number of observations to retain
          - obs_shape: tuple representing the shape of a single normalize observation
          - fill_value: value used to fill initial buffer tensors
        '''
        self.n=n
        self.obs_shape=obs_shape
        self.fill_value=fill_value 
        self.device=device
        self.reset()

    def reset(self):
        self.buffer = deque(maxlen=self.n)
        for _ in range(self.n):
            self.buffer.append(torch.full(self.obs_shape, fill_value=self.fill_value))
    
    def append(self, obs):
        'Normalize a raw observation and append to the history buffer'
        norm = self.normalize_obs(obs)
        self.buffer.append(norm)

    def get_tensor(self):
        '''
        Return a single tensor containing the observations in the buffer.
        Uses torch.stack on the torch.Tensors within the deque; most recent 
        observations will be at the end of the first index. 

        Returns: (Sequence x Channels x Pitch Length x Pitch Width)
        '''
        if self.device is not None:
            return torch.stack(tuple(self.buffer)).to(self.device)
        else:
            return torch.stack(tuple(self.buffer))

    def normalize_obs(self, obs):
        'Return the normalized pixel observation in the shape (Channels x Length x Width)'
        return torch.from_numpy(obs/255).T.float()

def play_round_with_history(env,
                            model, 
                            target_network,
                            device,
                            loss_fn, 
                            optimizer, 
                            sync_freq, 
                            replay_buffer,
                            history_length,
                            batch_size, 
                            epsilon=0.2,
                            gamma=0.98):
    '''
    Play a single round for a provided environment. Uses a target network, experience replay,
    and incorporates a history buffer for agents with memory. The model used will need to accept 
    the result of HistoryBuffer.get_tensor() instead of the observation on its own.
    '''
    history = HistoryBuffer(n=history_length, device=device)
    obs = env.reset() 
    history.append(obs)
    rew = 0
    this_match_losses = []
    this_match_reward = 0

    model.train()
    j = 0
    done=False
    while not done:
        j += 1
        this_obs = history.get_tensor()
        qval = model(this_obs)
        if random.random() < epsilon:
            action_ = env.action_space.sample()
        else:
            action_ = qval.argmax().item()

        # make the move
        obs2, rew2, done2, info2 = env.step(action_)
        history.append(obs2)
        next_obs = history.get_tensor()
        exp = (this_obs, action_, rew, next_obs, done)
        replay_buffer.append(exp)
        rew = rew2
        done = done2
        this_match_reward += rew

        if len(replay_buffer) > batch_size:
            minibatch = random.sample(replay_buffer, batch_size)
            state1_batch, action_batch, reward_batch, state2_batch, done_batch = list(zip(*minibatch))
            state1_batch = torch.stack(state1_batch).to(device)
            action_batch = torch.Tensor(action_batch).to(device)
            reward_batch = torch.Tensor(reward_batch).to(device)
            state2_batch = torch.stack(state2_batch).to(device)
            done_batch = torch.Tensor(done_batch).to(device)

            Q1 = model(state1_batch)
            with torch.no_grad():
                Q2 = target_network(state2_batch)
            Y = reward_batch + gamma * ((1 - done_batch)) * torch.max(Q2, dim=1)[0]
            X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(1)).squeeze()
            X = X.to(device)
            Y = Y.to(device)
            loss = loss_fn(X, Y.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            this_match_losses.append(loss.item())

            if j % sync_freq == 0:
                target_network.load_state_dict(model.state_dict())

        if done:
            break

    return {'losses':this_match_losses, 'reward':this_match_reward}