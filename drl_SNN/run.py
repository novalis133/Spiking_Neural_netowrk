# Main
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from s_policy import *
import torch
import torch.nn as nn
import torch.optim as optim
import gym

# Initialize environment
env = gym.make('CartPole-v1')
torch.manual_seed(0)

device = torch.device("cpu")

# Initialize SNN layers
lif1 = snn.Leaky(beta=0.9, init_hidden=True)
lif2 = snn.Leaky(beta=0.9, init_hidden=True, output=True)

# Setup policy and optimizers
s_policy = S_Policy()
policy_optimizer = optim.Adam(s_policy.parameters(), lr=1e-2)

BP_net = nn.Sequential(nn.Flatten(),
                    nn.Linear(10,32),
                    lif1,
                    nn.Linear(32, 2),
                    lif2).to(device)

snn_optimizer = optim.Adam(BP_net.parameters(), lr=1e-3)  # Notice the different learning rate

try:
    # Train the network
    scores = reinforce(BP_net, s_policy, policy_optimizer, snn_optimizer, env, device)
    print(f"Training completed. Final scores: {scores}")
finally:
    # Cleanup
    env.close()