import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.optim import Adam
from torch.distributions import Normal


def mlp(sizes, activation, output_activation=nn.Identity()):
    """
    @info: create a multilayer perpectron architecture
    """
    layers=[]
    for i in range(len(sizes)-1):
        act = activation if (i+2)< len(sizes) else output_activation
        layers += [nn.Linear(sizes[i], sizes[i+1]), act]
    
    return nn.Sequential(*layers)


class DoubleQNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_layer=(64,64), activation=nn.ReLU(), lr=1e-4):
        super().__init__()
        """
        @info: create two neural networks to predict Q(s,a). Consider, networks's input as torch.cat((obs, act))
        @input:
        ------
            - obs_dim: number of observable states
            - act_dim: number of actions
            - hidden_layer: number of hidden neurons
            - activation: nonlinear activation function  
        """
        # create Q(s,a) networks
        self.q1_net = mlp([obs_dim+act_dim]+list(hidden_layer)+[1], activation=activation)
        self.q2_net = mlp([obs_dim+act_dim]+list(hidden_layer)+[1], activation=activation)

        # create optimizer
        self.optimizer = Adam(self.parameters(), lr=lr)

    def forward(self, obs, act):
        return self.q1_net.forward(torch.cat((obs, act),dim=1)), self.q2_net.forward(torch.cat((obs, act),dim=1))



class GaussianPolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, action_space, hidden_layer=(64,64), lr=1e-4):
        super().__init__()
        """
        @info: create a neural networks to predict mean(mu) and standar-deviation(std). Consider, networks's input torch.tensor(obs)
        @input:
        ------
            - obs_dim: number of observable states
            - act_dim: number of actions
            - hidden_layer: number of hidden neurons
            - activation: nonlinear activation function  
        """        
        # create pi(s,a) network
        self.layer1 = nn.Linear(obs_dim, hidden_layer[0])
        self.layer2 = nn.Linear(hidden_layer[0], hidden_layer[1])

        self.mu_layer = nn.Linear(hidden_layer[1], act_dim)
        self.log_std_layer = nn.Linear(hidden_layer[1], act_dim)

        # create optimizer
        self.optimizer = Adam(self.parameters(), lr=lr)

        # represent from [-1 +1] to action space
        self.action_scale = torch.tensor((action_space.high - action_space.low)/2, dtype=torch.float32)
        self.action_bias = torch.tensor((action_space.high + action_space.low)/2, dtype=torch.float32)

    def forward(self, obs):
        """
        @info compute mean(mu) and log standard-deviation (std)
        """
        x = f.relu(self.layer2(f.relu(self.layer1(obs))))
        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x)

        # limit value of log_std
        log_std = torch.clamp(log_std, min=-20, max=2)

        return mu, log_std

    def predict_action(self, obs, deterministic=False):
        # predict mean and standard-deviation
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        # create Gaussian distribution
        dist = Normal(mu, std)
        # reparametrization trick: n = mu + eps*std ==> eps=N(0,1)
        n = dist.rsample()
        # squeased gaussian distribution
        act = torch.tanh(n)
        # logarithmic probability of action: log N(n,s) - sum log(1 - act^2)
        logprob = dist.log_prob(n) - torch.log(self.action_scale*(1-act.pow(2)) + 1e-6)
        #print(f"logprob: {logprob}")
        
        #print(f"mu:{mu.item():.3f}, std:{std.item():.3f}, n:{n.item():.3f}, act:{act.item():.3f}")
        if len(logprob.size())>1: # check if logprob is a batch
            logprob = logprob.sum(axis=1, keepdim=True)
        else:
            logprob = logprob.sum(axis=0)
        # represent action in environment action space
        if deterministic:
            return self.action_scale*mu + self.action_bias
        else:
            return self.action_scale*act + self.action_bias, logprob