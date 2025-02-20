# Description: Policy network for the SNN agent
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import Dataset, DataLoader
import snntorch as snn
import snntorch.functional as SF
from snntorch import backprop

# For visualization
import base64, io
from gym.wrappers.monitoring import video_recorder
from IPython.display import HTML
from IPython import display
import glob

# Set random seed for reproducibility
torch.manual_seed(0)

class S_Policy(nn.Module):
    def __init__(self, num_inputs=4, num_hidden=32, num_outputs=2):
        super().__init__()
        # Network Architecture
        beta = 0.95
        self.device = torch.device("cpu")
        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):
        x = x.to(self.device)
        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        cur1 = self.fc1(x)
        spk1, mem1 = self.lif1(cur1, mem1)
        cur2 = self.fc2(spk1)
        spk2, mem2 = self.lif2(cur2, mem2)
        spk2_rec.append(spk2)
        mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)

    def act(self, state, temperature=1.0):
        """
        This method selects an action based on the state.

        Args:
        - state: The current state of the environment
        - temperature (float, optional): Temperature parameter for the softmax function to control
        exploration-exploitation balance. It can be any positive real number, typically around 1.0.
        High temperature (greater than 1.0) leads to more exploration (actions have similar probability),
        and low temperature (less than 1.0) leads to more exploitation (the action with the highest
        original probability is more likely to be chosen).

        Returns:
        - action (int): The selected action.
        - action_dist.log_prob(action) (Tensor): The log probability of the selected action.
        """

        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device) # Prepare state for network input
        action_probs, _ = self.forward(state) # Get action probabilities
        action_probs = action_probs.squeeze(0)

        # Adjust action probabilities using temperature and create a categorical distribution
        action_dist = Categorical(F.softmax(action_probs / temperature, dim=-1))

        action = action_dist.sample() # Sample an action
        return action.item(), action_dist.log_prob(action) # Return the action and the log probability

class RL_Dataset(Dataset):
    def __init__(self, states, actions):
        self.states = states
        self.actions = actions

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        state = torch.tensor(self.states[idx]).float()
        target = torch.tensor(self.actions[idx]).long()
        return state, target

def reinforce(policy, BP_policy, policy_optimizer, snn_optimizer,
             env, device, n_episodes=500, max_t=10, gamma=1.0, print_every=100):
    """
    Train a policy using the REINFORCE algorithm.

    Parameters:
    policy (Policy): The policy to train.
    optimizer (torch.optim.Optimizer): The optimizer to use for training the policy.
    n_episodes (int, optional): The maximum number of training episodes. Default is 1000.
    max_t (int, optional): The maximum number of timesteps per episode. Default is 1000.
    gamma (float, optional): The discount factor. Default is 1.0.
    print_every (int, optional): How often to print average score. Default is 100.

    Returns:
    scores (List[float]): A list of scores from each episode of the training. The score is the total reward obtained in the episode.
    """

    # Create a double-ended queue to hold the most recent 100 episode scores
    scores_deque = deque(maxlen=100)

    # List to store all episode scores
    scores = []
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []
    loss_fn = SF.mse_count_loss()
    reg_fn = SF.l1_rate_sparsity()
    # Loop over each episode
    for i_episode in range(1, n_episodes+1):
        # List to save log probabilities for each step of this episode
        saved_log_probs = []

        # List to save rewards for each step of this episode
        episode_rewards = []

        # Reset the environment and get initial state
        state = env.reset(seed=0)

        # Collect trajectory
        for t in range(max_t):
            # Use the policy to select an action given the current state
            action, log_prob = policy.act(state)

            # Save the log probability of the chosen action
            saved_log_probs.append(log_prob)

            # Take the action and get the new state and reward
            state_, reward, done, _ = env.step(action)

            # Add the reward to the list of rewards for this episode
            episode_rewards.append(reward)
            # Store experience
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(state_)
            dones.append(done)
            state = state_
            # If the episode is done, break out of the loop
            if done:
                break

        # Calculate total reward for this episode and add it to the deque and list of scores
        scores_deque.append(sum(episode_rewards))
        scores.append(sum(episode_rewards))

        # Compute future discount rewards for each step
        discounts = [gamma**i for i in range(len(episode_rewards)+1)]

        # Calculate total discounted reward for the episode
        R = sum([a*b for a, b in zip(discounts, episode_rewards)])

        # Compute the policy loss
        policy_loss = []
        for log_prob in saved_log_probs:
            policy_loss.append(-log_prob * R) # note that gradient ascent is the same as gradient descent with negative rewards
        policy_loss = torch.cat(policy_loss).sum()
        # Creating the dataset
        dataset = RL_Dataset(states, actions)
        # Create a dataloader
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
        # Backprobagate trhoough time
        loss= backprop.BPTT(BP_policy, dataloader, optimizer=snn_optimizer,
                             criterion=loss_fn, num_steps=max_t, time_var=False,
                             regularization=reg_fn, device=device)


        # Perform a step of policy gradient descent
        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        # Clear the computation graph
        #torch.cuda.empty_cache()

        # Perform a step of SNN optimization
        #snn_optimizer.zero_grad()
        #loss.backward()
        #snn_optimizer.step()


        # Print current average score every 'print_every' episodes
        if i_episode % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}\tSNN Loss Score: {:.2f}\tPolicy Loss Score: {:.2f}'.format(i_episode,
                                                                                                                np.mean(scores_deque),
                                                                                                                loss,
                                                                                                                policy_loss))

        # Stop if the environment is solved
        if np.mean(scores_deque)>=500.0:
            print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_deque)))
            break


    # Return all episode scores
    return scores