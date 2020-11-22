'''
Utility file to be zipped/uploaded with the model weights and main.py. 
Everything in the zipfile will be extracted into /kaggle_simulations/agent,
but only main.py gets moved into the execution location. Thus, we can load 
the weights using relative paths here, but this script will need to be loaded
into main.py after doing `sys.path.append('/kaggle_simulations/agent')`.
'''

import torch
from torch import nn
from torch.nn import functional as F
from collections import deque
from pathlib import Path
import numpy as np
import os

MODEL_DIR = Path(__file__).resolve().parent
MODEL_NAME = '20201122-torchbeast-model.tar'
HISTORY_LENGTH = 4

# ----- data utils -----
# my history buffer class
class HistoryBuffer():
    def __init__(self, n, obs_shape=(4, 84, 84), fill_value=0., device=None):
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
            return torch.cat(list(self.buffer)).to(self.device)
        else:
            return torch.cat(list(self.buffer))

    def normalize_obs(self, obs):
        'Return the normalized pixel observation in the shape (Channels x Length x Width)'
        return torch.from_numpy((obs/255).transpose(2, 0, 1)).float()

# ----- model code -----
class AtariNet(nn.Module):
    def __init__(self, observation_shape, num_actions, use_lstm=False):
        super(AtariNet, self).__init__()
        self.observation_shape = observation_shape
        self.num_actions = num_actions

        # Feature extraction.
        self.conv1 = nn.Conv2d(
            in_channels=self.observation_shape[0],
            out_channels=32,
            kernel_size=8,
            stride=4,
        )
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Fully connected layer.
        self.fc = nn.Linear(3136, 512)

        # FC output size + one-hot of last action + last reward.
        core_output_size = self.fc.out_features + num_actions + 1

        self.use_lstm = use_lstm
        if use_lstm:
            self.core = nn.LSTM(core_output_size, core_output_size, 2)

        self.policy = nn.Linear(core_output_size, self.num_actions)
        self.baseline = nn.Linear(core_output_size, 1)

    def initial_state(self, batch_size):
        if not self.use_lstm:
            return tuple()
        return tuple(
            torch.zeros(self.core.num_layers, batch_size, self.core.hidden_size)
            for _ in range(2)
        )

    def forward(self, inputs, core_state=()):
        x = inputs["frame"]  # [T, B, C, H, W].
        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        x = x.float() / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(T * B, -1)
        x = F.relu(self.fc(x))

        one_hot_last_action = F.one_hot(
            inputs["last_action"].view(T * B), self.num_actions
        ).float()
        clipped_reward = torch.clamp(inputs["reward"], -1, 1).view(T * B, 1)
        core_input = torch.cat([x, clipped_reward, one_hot_last_action], dim=-1)

        if self.use_lstm:
            core_input = core_input.view(T, B, -1)
            core_output_list = []
            notdone = (~inputs["done"]).float()
            for input, nd in zip(core_input.unbind(), notdone.unbind()):
                # Reset core state to zero whenever an episode ended.
                # Make `done` broadcastable with (num_layers, B, hidden_size)
                # states:
                nd = nd.view(1, -1, 1)
                core_state = tuple(nd * s for s in core_state)
                output, core_state = self.core(input.unsqueeze(0), core_state)
                core_output_list.append(output)
            core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        else:
            core_output = core_input
            core_state = tuple()

        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)

        if self.training:
            action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            # Don't sample when testing.
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        return (
            dict(policy_logits=policy_logits, baseline=baseline, action=action),
            core_state,
        )
    
model = AtariNet((16, 84, 84), num_actions=19, use_lstm=True)
checkpoint = torch.load(os.path.join(MODEL_DIR, MODEL_NAME), map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

def get_history(length=HISTORY_LENGTH):
    return HistoryBuffer(length)

def get_model():
    '''
    This one takes in a dictionary with information like last action, reward, etc. - unsure if reward
    exists in the raw observation. I'll either constantly use 0 or derive something to keep track of 
    the running score.
    
    Outputs a tuple of (dictionary containing results, unused final hidden states)
    '''
    return model

if __name__ == '__main__':
    model = get_model()
    print(model)