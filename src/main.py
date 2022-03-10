import gym
import gym_carla

import random
import datetime
import numpy as np
from tqdm import tqdm
from pathlib import Path
from agent_sac import Agent
from metrics import MetricLogger
from playground import PlayGround


N_ACTIONS               = 6
INPUT_SHAPE             = (9,256,256)
REPLAY_SIZE             = 100000
BATCH_SIZE              = 128
ALPHA                   = 0.00025
EPSILON                 = 1.0
EPSILON_DACAY           = 0.99999965 
EPISODES                = 100000
BURNIN                  = 10000
SAVE_EVERY              = 30000
UPDATE_EVERY            = 5000
TRAIN_EVERY             = 3
DISCOUNT                = 0.99



def main():

    params = {
        'number_of_vehicles': 20,
        'number_of_walkers': 0,
        'display_size': 256,  # screen size of bird-eye render
        'max_past_step': 1,  # the number of past steps to draw
        'dt': 0.05,  # time interval between two frames
        'discrete': False,  # whether to use discrete control space
        'discrete_acc': [-0.5, 0.0, 0.3, 0.6],  # discrete value of accelerations
        'discrete_steer': [-1.0, -0.5, 0.0, 0.5, 1.0],  # discrete value of steering angles
        'continuous_accel_range': [-1.0, 1.0],  # continuous acceleration range
        'continuous_steer_range': [-1.0, 1.0],  # continuous steering angle range
        'ego_vehicle_filter': 'vehicle.model3.*',  # filter for defining ego vehicle
        'port': 2000,  # connection port
        'town': 'Town03',  # which town to simulate
        'task_mode': 'random',  # mode of the task, [random, roundabout (only for Town03)]
        'max_time_episode': 500,  # maximum timesteps per episode
        'max_waypt': 12,  # maximum number of waypoints
        'obs_range': 32,  # observation range (meter)
        'lidar_bin': 0.125,  # bin size of lidar sensor (meter)
        'd_behind': 12,  # distance behind the ego vehicle (meter)
        'out_lane_thres': 2.0,  # threshold for out of lane
        'desired_speed': 8,  # desired speed (m/s)
        'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
        'display_route': True,  # whether to render the desired route
        'pixor_size': 64,  # size of the pixor labels
        'pixor': False,  # whether to output PIXOR observation
        }

    save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    save_dir.mkdir(parents=True)
    checkpoint = Path('online_16.chkpt')
    random.seed(1)
    np.random.seed(1)

    # Instantiate Environments
    # Set gym-carla environment
    env = gym.make('carla-v0', params=params)

    tesla = Agent(
        alpha=ALPHA, 
        gamma=DISCOUNT, 
        epsilon=EPSILON, 
        tau=1,
        epsilon_decay=EPSILON_DACAY,
        max_size=REPLAY_SIZE, 
        burnin=BURNIN,
        input_shape=INPUT_SHAPE, 
        action_space=env.action_space,
        batch_size=BATCH_SIZE, 
        save_every=SAVE_EVERY, 
        update_every=UPDATE_EVERY,
        train_every=TRAIN_EVERY, 
        save_dir=save_dir, 
        checkpoint=checkpoint
    )
    
    logger = MetricLogger(save_dir=save_dir)

    for epi in tqdm(range(1,EPISODES + 1), ascii=True, unit='episodes'):
        
        # Restarting episode - reset episode reward and step number
        episode_reward = 0
        state = env.reset()

        try:
        
            while True:
                action = tesla.act(state)
                next_state, reward, done, _ = env.step(action)
                tesla.cache(state, action, reward, next_state, done)
                
                q,loss = tesla.learn()
                logger.log_step(reward, loss, q)

                state = next_state
                # print(reward, loss)

                if done:
                    break
        finally:
            # env.destroy() # no attribute destroy 
            env.reset()

        logger.log_episode()

        if epi % 20 == 0:
            logger.record(
                episode=epi,
                epsilon=tesla.epsilon,
                step=tesla.curr_step
            )
        
        if env.to_quit():
            env.set_synchronous_mode(False)
            break


if __name__ == '__main__':
    main()
