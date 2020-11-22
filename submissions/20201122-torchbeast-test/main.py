import sys
from gfootball.env import observation_preprocessing
RESOURCE_DIR = '/kaggle_simulations/agent'
sys.path.append(RESOURCE_DIR)
import model_prep as mp

model = mp.get_model()
model.eval()
history = mp.get_history()

# function to convert kaggle's passed obs to the pixel representation
def obs_convert(obs):
    return observation_preprocessing.generate_smm([obs['players_raw'][0]], channel_dimensions=(84, 84))[0]

last_action = 0
initial_state = model.initial_state(1)

def agent(obs):
    global model
    global history
    global last_action
    
    frame = obs_convert(obs)
    history.append(frame)
    model_obs = history.get_tensor().unsqueeze(0).unsqueeze(0)
    model_input = {
        'frame':model_obs,
        'last_action':torch.tensor(last_action),
        'reward':torch.tensor(0).view(1, 1),
        'done':torch.tensor(False).view(1, 1)
    }
    model_output, _ = model(model_input, initial_state)
    action = int(model_output['action'].item())
    
    last_action = action
    
    return [int(action)]