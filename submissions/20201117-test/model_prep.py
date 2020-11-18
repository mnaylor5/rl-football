'''
Utility file to be zipped/uploaded with the model weights and main.py. 
Everything in the zipfile will be extracted into /kaggle_simulations/agent,
but only main.py gets moved into the execution location. Thus, we can load 
the weights using relative paths here, but this script will need to be loaded
into main.py after doing `sys.path.append('/kaggle_simulations/agent')`.
'''

import torch
from torch import nn
from collections import deque
from pathlib import Path
import os

MODEL_DIR = Path(__file__).resolve().parent
MODEL_NAME = '20201117-cnn_lstm_dqn_single_goal_lazy_300k.pt'
HISTORY_LENGTH = 5

# ----- data utils -----
# my history buffer class
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

# ----- model code -----
# encoder for pixel images
class CNNEncoder(nn.Module):
    def __init__(self, out_size):
        super(CNNEncoder, self).__init__()
        self.c1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=1)
        self.c2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=1)
        self.c3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.c4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.LeakyReLU()
        self.linear = nn.Linear(1536, out_size)
        
    def forward(self, x):
        h = self.relu(self.c1(x))
        h = self.relu(self.c2(h))
        h = self.relu(self.c3(h))
        h = self.relu(self.c4(h))
        flattened = h.flatten(-3)
        out = self.relu(self.linear(flattened))
        return out

# agent class
class HistoryConvAgent(nn.Module):
    def __init__(self, dropout_p = 0.1, action_size=18):
        super(HistoryConvAgent, self).__init__()
        self.encoder = CNNEncoder(out_size=256)
        self.gru = nn.GRU(256, 256, num_layers=1, bidirectional=False, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, action_size)
        self.activation = nn.LeakyReLU()
        
    def forward(self, x):
        # batching doesn't play nicely here
        if x.ndim == 4:
            encoded = self.encoder(x)
            _, gru_out = self.gru(encoded.unsqueeze(0))
        else:
            encoded = torch.stack([self.encoder(x[i]) for i in range(x.shape[0])])
            _, gru_out = self.gru(encoded)
        gru_out = self.dropout(gru_out.squeeze())
        fc1_out = self.activation(self.fc1(gru_out))
        fc2_out = self.activation(self.fc2(fc1_out))
        return fc2_out
    
# load model 
dqn = HistoryConvAgent()
dqn.load_state_dict(torch.load(os.path.join(MODEL_DIR, MODEL_NAME)))
dqn.eval()

def get_history(length=HISTORY_LENGTH):
    return HistoryBuffer(length)

def get_model():
    return dqn