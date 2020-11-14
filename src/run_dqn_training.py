'''
Train a CNN-LSTM DQN agent with a curriculum on the football problem.
'''
import torch 
from torch import nn
import training_ground as tg 
import dqn_utils as dq 
from collections import defaultdict, deque
import tqdm
from copy import deepcopy
import numpy as np

# build the model
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
    def __init__(self, dropout_p = 0.1):
        super(HistoryConvAgent, self).__init__()
        self.encoder = CNNEncoder(out_size=256)
        self.gru = nn.GRU(256, 256, num_layers=1, bidirectional=False, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 18)
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

if __name__ == '__main__':
    plan = tg.TrainingPlan(basic_rounds=10, 
                           easy_rounds=10, 
                           medium_rounds=10, 
                           hard_rounds=10, 
                           full_match_rounds=10)
    progress = tqdm.tqdm(range(len(plan.training_plan)))
    model = HistoryConvAgent()
    target_net = deepcopy(model)
    loss_fn = nn.MSELoss()
    optim=torch.optim.RMSprop(model.parameters())
    replay_buffer = deque(maxlen=128)
    rewards = defaultdict(list)
    for match in progress:
        env = plan.get_next()
        progress.set_description(plan.current_scenario_name)
        performance = dq.play_round_with_history(
            env,
            model=model, 
            target_network=target_net,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            loss_fn = loss_fn, # need
            optimizer=optim, # need
            sync_freq=1_000,
            replay_buffer=replay_buffer, # need
            history_length=10,
            batch_size=64,
            epsilon=0.4, # figure out / refine
            gamma=0.999
        )

        rewards[plan.current_scenario_name].append(performance['reward'])
    
    for s, r in rewards.items():
        if r != []:
            print(f"{s}: {np.mean(r):.4}")
