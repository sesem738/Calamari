import os
import time
from tqdm import tqdm
import random
import datetime
import numpy as np
from pathlib import Path
from agent import Agent
from model import Network
from metrics import MetricLogger
from playground import PlayGround



if __name__ == '__main__':

    # Create models folder
    if not os.path.isdir('models'):
        print('Model directory not found')
    
    # Log directory
    save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    save_dir.mkdir(parents=True)

    # Enter checkpoint path
    checkpoint = Path('net_155.chkpt')

    episodes = 100


    random.seed(1)
    np.random.seed(1)

    # Instantiate Environments
    tesla = Agent(state_dim=(3,480,640), action_dim=6, save_dir=save_dir, checkpoint=checkpoint)
    env = PlayGround()
    logger = MetricLogger(save_dir=save_dir)


    for epi in tqdm(range(1,episodes + 1), ascii=True, unit='episodes'):
        
        # Restarting episode - reset episode reward and step number
        episode_reward = 0
        state = env.reset()

        try:
        
            while True:
                action = tesla.act(state)
                next_state, reward, done, _ = env.step(action)
                # tesla.cache(state, next_state, action, reward, done)
                # logger.log_step(reward, None, None)
                state = next_state

                if done:
                    break
        
        finally:
        
            env.destroy()

        # Quit simulation after last episode if pygame screen closed
        if env.to_quit():
            break
    
    # logger.log_episode()

    # if epi % 20 == 0:
    #     logger.record(
    #         episode=epi,
    #         epsilon=tesla.exploration_rate,
    #         step=tesla.curr_step
    #     )




