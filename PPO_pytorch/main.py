import argparse
import gym
import sys
import torch

from ppo import PPO
from network import FeedForwardNN

def get_args():

	parser = argparse.ArgumentParser()

	parser.add_argument('--mode', dest='mode', type=str, default='train')              
	parser.add_argument('--actor_model', dest='actor_model', type=str, default='')    
	parser.add_argument('--critic_model', dest='critic_model', type=str, default='')  

	args = parser.parse_args()

	return args

def train(env, hyperparameters, actor_model, critic_model):
	
	print("Training")

	# Create a model for PPO.
	model = PPO(policy_class=FeedForwardNN, env=env, **hyperparameters)

	# Tries to load in an existing actor/critic model to continue training on
	if actor_model != '' and critic_model != '':
		model.actor.load_state_dict(torch.load(actor_model))
		model.critic.load_state_dict(torch.load(critic_model))
	else:
		print("Training from scratch")

	model.learn(total_timesteps=200_00)# original code used 200_000_000

def test(env, actor_model):

	print("Testing {actor_model}")

	obs_dim = env.observation_space.shape[0]
	act_dim = env.action_space.shape[0]
    
	policy = FeedForwardNN(obs_dim, act_dim)

	policy.load_state_dict(torch.load(actor_model))

def main(args):
	hyperparameters = {
				'timesteps_per_batch': 2048, 
				'max_timesteps_per_episode': 200, 
				'gamma': 0.99, 
				'n_updates_per_iteration': 10,
				'lr': 3e-4, 
				'clip': 0.2,
				'render': True,
				'render_every_i': 10
			  }
	env = gym.make('Pendulum-v0')

	if args.mode == 'train':
		train(env=env, hyperparameters=hyperparameters, actor_model=args.actor_model, critic_model=args.critic_model)
	else:
		test(env=env, actor_model=args.actor_model)

if __name__ == '__main__':
	args = get_args() 
	main(args)
