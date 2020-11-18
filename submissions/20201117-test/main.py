import sys
from gfootball.env import observation_preprocessing
RESOURCE_DIR = '/kaggle_simulations/agent'
sys.path.append(RESOURCE_DIR)
import model_prep as mp

dqn = mp.get_model()
dqn.eval()
history = mp.get_history()

# function to convert kaggle's passed obs to the pixel representation
def obs_convert(obs):
    return observation_preprocessing.generate_smm([obs['players_raw'][0]])[0]

def agent(obs):
    global dqn
    global history
    
    clean_obs = obs_convert(obs)
    history.append(clean_obs)
    qval = dqn(history.get_tensor())
    action = qval.argmax().item()
    
    return [int(action)]