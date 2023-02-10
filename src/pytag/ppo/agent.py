import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

class ActorCritic(nn.Module):
    def __init__(self, args, input_dims, n_actions):
        super(ActorCritic, self).__init__()
        self.args = args
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.hidden_units = args.hidden_size

        self.network = nn.Sequential(nn.Linear(input_dims, self.hidden_units), nn.ReLU(), nn.Linear(self.hidden_units, self.hidden_units), nn.ReLU())

        self.actor = nn.Linear(self.hidden_units, self.n_actions)
        self.critic = nn.Linear(self.hidden_units, 1)

    def forward(self, x):
        x_ = self.network(x)
        return x_

    def act(self, obs):
        x_ = self(obs)

        action_probs = self.actor(x_)
        action_probs = F.softmax(action_probs, dim=-1)
        dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def get_logprobs(self, obs):
        x_ = self(obs)

        action_probs = self.actor(x_)
        action_probs = F.softmax(action_probs, dim=-1)

        return action_probs.detach()

    def evaluate(self, obs, action):
        x_ = self(obs)
        action_probs = self.actor(x_)
        action_probs = F.softmax(action_probs, dim=-1)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(x_)

        return action_logprobs, state_values, dist_entropy
class PPO:
    def __init__(self, args, env, model=None):
        self.name = "PPO"
        self.random = np.random.RandomState(args.seed)
        self.args = args
        self.device = args.device
        self.env = env

        self.args = args
        self.gamma = args.gamma

        self.lr = args.learning_rate
        self.lr_actor = args.lr_actor
        self.lr_critic = args.lr_critic

        self.eps_clip = args.eps_clip
        self.K_epochs = args.k_epochs

        self.n_actions = env.action_space()
        self.input_dims = env.observation_space()

        self.policy = ActorCritic(args, self.input_dims, self.n_actions).to(args.device)
        if model is None:
            model = args.model
        if model:  # Load pretrained model if provided
            if os.path.isfile(model):
                checkpoint = torch.load(model,
                                        map_location='cpu')  # Always load tensors onto CPU by default, will shift to GPU if necessary
                self.policy.load_state_dict(checkpoint['state_dict'])
                print("Loading pretrained model: " + model)
            else:  # Raise error if incorrect model path provided
                raise FileNotFoundError(model)

        self.optim = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': self.lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': self.lr_critic}
        ])
        # self.optim = torch.optim.Adam(self.policy.parameters(), lr=args.learning_rate)

        self.policy_old = ActorCritic(args, self.input_dims, self.n_actions, self.scalar_dims).to(self.args.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def act(self, state):
        with torch.no_grad():
            action, action_logprob = self.policy_old.act(state)

        return action.item(), action_logprob.detach()

    def get_logprobs(self, state):
        with torch.no_grad():
            logprobs = self.policy.get_logprobs(state)
        return logprobs

    def learn(self, memory):
        transitions = memory.get_buffer()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(transitions[0], transitions[1].squeeze())

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - transitions[2].detach())

            # Finding Surrogate Loss
            advantages = transitions[3] - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, transitions[3]) - 0.01 * dist_entropy

            # take gradient step
            self.optim.zero_grad()
            loss.mean().backward()
            self.optim.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # reset buffer
        memory.reset()

        return loss.mean().cpu().detach().numpy()

    def save_model(self, path, name="checkpoint.pth"):
        state = {
            "state_dict": self.policy.state_dict(),
            "optimizer": self.optim.state_dict(),
            "args": self.args,
            "gamma": self.gamma,
        }
        torch.save(state, os.path.join(path, name))
        pass

    def load_model(self, path, eval_only=False):

        state = torch.load(path)
        self.args = state["args"]
        self.policy = ActorCritic(self.args, self.input_dims, self.n_actions, self.scalar_dims)
        self.policy.load_state_dict(state["state_dict"])

        if eval_only:
            self.policy.eval()
        else:
            # load and copy q_net params to target_net
            self.old_policy = ActorCritic(self.args, self.input_dims, self.n_actions, self.scalar_dims)
            self.old_policy.load_state_dict(state["state_dict"])
            self.optim.load_state_dict(state["optimizer"])
            self.gamma = state["gamma"]



