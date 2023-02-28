"""
CQL-GNN
-------
This file contains the CQL-GNN specifications. In particular, we implement:
(1) GNNParser
    Converts raw environment observations to agent inputs (s_t).
(2) GNNActor:
    Policy parametrized by Graph Convolution Networks 
(3) GNNCritic:
    Critic parametrized by Graph Convolution Networks
(4) SAC+CQL:
    Soft actor critic using a GNN parametrization for both Actor and Critic and the conservative loss of CQL 
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Dirichlet
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv
from src.algos.reb_flow_solver import solveRebFlow
from src.misc.utils import dictsum
import json


class Scalar(nn.Module):
    def __init__(self, init_value):
        super().__init__()
        self.constant = nn.Parameter(
            torch.tensor(init_value, dtype=torch.float32))

    def forward(self):
        return self.constant


class GNNParser():
    """
    Parser converting raw environment observations to agent inputs (s_t).
    """

    def __init__(self, env, T=10, json_file=None, scale_factor=0.01):
        super().__init__()
        self.env = env
        self.T = T
        self.s = scale_factor
        self.json_file = json_file
        if self.json_file is not None:
            with open(json_file, "r") as file:
                self.data = json.load(file)

    def parse_obs(self, obs, device):
        x = torch.cat((
            torch.tensor([obs[0][n][self.env.time+1]*self.s for n in self.env.region]
                         ).view(1, 1, self.env.nregion).float(),
            torch.tensor([[(obs[0][n][self.env.time+1] + self.env.dacc[n][t])*self.s for n in self.env.region]
                          for t in range(self.env.time+1, self.env.time+self.T+1)]).view(1, self.T, self.env.nregion).float(),
            torch.tensor([[sum([(self.env.scenario.demand_input[i, j][t])*(self.env.price[i, j][t])*self.s
                          for j in self.env.region]) for i in self.env.region] for t in range(self.env.time+1, self.env.time+self.T+1)]).view(1, self.T, self.env.nregion).float()),
                      dim=1).squeeze(0).view(1+self.T + self.T, self.env.nregion).T
        if self.json_file is not None:
            edge_index = torch.vstack((torch.tensor([edge['i'] for edge in self.data["topology_graph"]]).view(1, -1),
                                      torch.tensor([edge['j'] for edge in self.data["topology_graph"]]).view(1, -1))).long()
        else:
            edge_index = torch.cat((torch.arange(self.env.nregion).view(1, self.env.nregion),
                                    torch.arange(self.env.nregion).view(1, self.env.nregion)), dim=0).long()
        data = Data(x, edge_index).to(device)
        return data

#########################################
############## ACTOR ####################
#########################################


class GNNActor(nn.Module):
    """
    Actor \pi(a_t | s_t) parametrizing the concentration parameters of a Dirichlet Policy.
    """

    def __init__(self, in_channels, hidden_size=32, act_dim=6):
        super().__init__()
        self.in_channels = in_channels
        self.act_dim = act_dim
        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, 1)

    def forward(self, state, edge_index, deterministic=False):
        out = F.relu(self.conv1(state, edge_index))
        x = out + state
        x = x.reshape(-1, self.act_dim, self.in_channels)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.softplus(self.lin3(x))
        concentration = x.squeeze(-1) + 1e-20
        if deterministic:
            action = (concentration) / (concentration.sum() + 1e-20)
            log_prob = None
        else:
            m = Dirichlet(concentration)
            action = m.rsample()
            log_prob = m.log_prob(action)

        return action, log_prob

#########################################
############## CRITIC ###################
#########################################


class GNNCritic(nn.Module):
    """
    Critic parametrizing the value function estimator V(s_t).
    """

    def __init__(self, in_channels, hidden_size=32, act_dim=16):
        super().__init__()
        self.act_dim = act_dim
        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels + 1, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, 1)
        self.in_channels = in_channels

    def forward(self, state, edge_index, action):
        out = F.relu(self.conv1(state, edge_index))
        x = out + state
        x = x.reshape(-1, self.act_dim, self.in_channels)
        concat = torch.cat([x, action.unsqueeze(-1)], dim=-1)
        x = F.relu(self.lin1(concat))
        x = F.relu(self.lin2(x))  # (B, N, H)
        x = torch.sum(x, dim=1)  # (B, H)
        x = self.lin3(x).squeeze(-1)  # (B, 1)
        return x

#########################################
############## A2C AGENT ################
#########################################


class SAC(nn.Module):
    """
    Advantage Actor Critic algorithm for the AMoD control problem.
    """

    def __init__(
        self,
        env,
        input_size,  # state dimension
        hidden_size=32,  # hidden units in MLP
        alpha=0.2,  # entropy coefficent
        gamma=0.97,  # discount factor
        polyak=0.995,  # polyak averaging
        p_lr=3e-4,  # actor learning rate
        q_lr=1e-3,  # critic learning rate
        use_automatic_entropy_tuning=False,
        n=1,  # n-step return backup for Q-function
        # lagrange threshold tau for automatic tuning of conservative weight eta
        lagrange_thresh=-1,
        min_q_weight=10,  # conservative weight eta
        deterministic_backup=True,  # determinsitic backup of the q-function
        device=torch.device("cpu"),
        min_q_version=3,  # version 2: CQL(rho), version 3: CQL(H)
        clip=200,
        json_file=None,  # data file for parser
    ):
        super(SAC, self).__init__()
        self.env = env
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device

        # SAC parameters
        self.alpha = alpha
        self.polyak = polyak
        self.env = env
        self.p_lr = p_lr
        self.q_lr = q_lr
        self.gamma = gamma
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        self.min_q_version = min_q_version
        self.act_dim = self.env.nregion
        self.n = n

        # conservative Q learning parameters
        self.num_random = 10
        self.temp = 1.0
        self.clip = clip
        self.min_q_weight = min_q_weight
        if lagrange_thresh == -1:
            self.with_lagrange = False
        else:
            print("using lagrange")
            self.with_lagrange = True
        self.deterministic_backup = deterministic_backup

        # nnets
        self.actor = GNNActor(self.input_size, self.hidden_size,
                              act_dim=self.act_dim).to(self.device)
        print(self.actor)
        self.critic1 = GNNCritic(
            self.input_size, self.hidden_size, act_dim=self.act_dim).to(self.device)
        self.critic2 = GNNCritic(
            self.input_size, self.hidden_size, act_dim=self.act_dim).to(self.device)
        assert self.critic1.parameters() != self.critic2.parameters()
        print(self.critic1)

        self.critic1_target = GNNCritic(
            self.input_size, self.hidden_size, act_dim=self.act_dim).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target = GNNCritic(
            self.input_size, self.hidden_size, act_dim=self.act_dim).to(self.device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        for p in self.critic1_target.parameters():
            p.requires_grad = False
        for p in self.critic2_target.parameters():
            p.requires_grad = False

        self.obs_parser = GNNParser(self.env, json_file=json_file, T=6)
        if self.wandb != None:
            self.wandb.watch(self.actor, log_freq=60)
            # self.wandb.watch(self.critic1, log_freq=60)

        self.optimizers = self.configure_optimizers()

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []
        self.to(self.device)

        if self.with_lagrange:
            self.target_action_gap = lagrange_thresh  # lagrange treshhold
            self.log_alpha_prime = Scalar(0.0).to(self.device)
            self.alpha_prime_optimizer = torch.optim.Adam(
                self.log_alpha_prime.parameters(), lr=self.p_lr)
            self.min_q_weight = 1.0

        if self.use_automatic_entropy_tuning:
            self.target_entropy = -np.prod(self.act_dim).item()
            self.log_alpha = Scalar(0.0)
            self.alpha_optimizer = torch.optim.Adam(
                self.log_alpha.parameters(), lr=self.p_lr
            )

    def parse_obs(self, obs, device):
        state = self.obs_parser.parse_obs(obs, device)
        return state

    def select_action(self, state, edge_index, deterministic=False):
        with torch.no_grad():
            a, _ = self.actor(state, edge_index, deterministic)
        a = a.squeeze(-1)
        a = a.detach().cpu().numpy()[0]
        return list(a)

    def compute_loss_q(self, data, conservative=False):

        (state_batch,
         edge_index,
         next_state_batch,
         edge_index2,
         reward_batch,
         action_batch) = data.x_s, data.edge_index_s, data.x_t, data.edge_index_t, data.reward, data.action.reshape(-1, self.env.nregion)

        q1 = self.critic1(state_batch, edge_index, action_batch)
        q2 = self.critic2(state_batch, edge_index, action_batch)
        with torch.no_grad():
            # Target actions come from current policy
            a2, logp_a2 = self.actor(next_state_batch, edge_index2)
            q1_pi_targ = self.critic1_target(next_state_batch, edge_index2, a2)
            q2_pi_targ = self.critic2_target(next_state_batch, edge_index2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)

            if not self.deterministic_backup:
                backup = reward_batch + self.gamma * \
                    (q_pi_targ - self.alpha * logp_a2)
            else:
                if self.n > 1:
                    backup = reward_batch + (self.gamma**self.n) * (q_pi_targ)
                else:
                    backup = reward_batch + self.gamma * q_pi_targ

        loss_q1 = F.mse_loss(q1, backup)
        loss_q2 = F.mse_loss(q2, backup)

        if conservative:
            batch_size = action_batch.shape[0]
            action_dim = action_batch.shape[-1]

            random_log_prob, current_log, next_log, q1_rand, q2_rand, q1_current, q2_current, q1_next, q2_next = self._get_action_and_values(
                data, 10, batch_size, action_dim)

            if self.min_q_version == 2:
                cat_q1 = torch.cat(
                    [q1_rand, q1.unsqueeze(1).unsqueeze(
                        1), q1_next, q1_current], 1
                )
                cat_q2 = torch.cat(
                    [q2_rand, q2.unsqueeze(1).unsqueeze(
                        1), q2_next, q2_current], 1
                )

            if self.min_q_version == 3:
                # importance sampled version
                cat_q1 = torch.cat([q1_rand - random_log_prob, q1_next -
                                   next_log.detach(), q1_current - current_log.detach(),], 1,)
                cat_q2 = torch.cat([q2_rand - random_log_prob, q2_next -
                                   next_log.detach(), q2_current - current_log.detach(),], 1,)

            min_qf1_loss = (torch.logsumexp(cat_q1 / self.temp,
                            dim=1,).mean() * self.min_q_weight * self.temp)
            min_qf2_loss = (torch.logsumexp(cat_q2 / self.temp,
                            dim=1,).mean() * self.min_q_weight * self.temp)

            """Subtract the log likelihood of data"""
            min_qf1_loss = min_qf1_loss - q1.mean() * self.min_q_weight
            min_qf2_loss = min_qf2_loss - q2.mean() * self.min_q_weight

            if self.with_lagrange:
                alpha_prime = torch.clamp(
                    torch.exp(self.log_alpha_prime()), min=0.0, max=1000000.0
                )
                min_qf1_loss = alpha_prime * \
                    (min_qf1_loss - self.target_action_gap)
                min_qf2_loss = alpha_prime * \
                    (min_qf2_loss - self.target_action_gap)

                self.alpha_prime_optimizer.zero_grad()
                alpha_prime_loss = (-min_qf1_loss - min_qf2_loss) * 0.5
                alpha_prime_loss.backward(retain_graph=True)
                self.alpha_prime_optimizer.step()

            loss_q1 = loss_q1 + min_qf1_loss
            loss_q2 = loss_q2 + min_qf2_loss

        return loss_q1, loss_q2

    def compute_loss_pi(self, data):
        state_batch, edge_index = data.x_s, data.edge_index_s
        actions, logp_a = self.actor(state_batch, edge_index)
        q1_1 = self.critic1(state_batch, edge_index, actions)
        q2_a = self.critic2(state_batch, edge_index, actions)
        q_a = torch.min(q1_1, q2_a)

        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha() * (logp_a +
                           self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha().exp()

        loss_pi = (self.alpha * logp_a - q_a).mean()

        return loss_pi

    def update(self, data, conservative=False):

        loss_q1, loss_q2 = self.compute_loss_q(data, conservative)

        self.optimizers["c1_optimizer"].zero_grad()
        loss_q1.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.critic1.parameters(),  self.clip)
        self.optimizers["c1_optimizer"].step()

        self.optimizers["c2_optimizer"].zero_grad()
        loss_q2.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.critic2.parameters(),  self.clip)
        self.optimizers["c2_optimizer"].step()

        for p in self.critic1.parameters():
            p.requires_grad = False
        for p in self.critic2.parameters():
            p.requires_grad = False

        self.optimizers["a_optimizer"].zero_grad()
        loss_pi = self.compute_loss_pi(data)
        loss_pi.backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.actor.parameters(), 10)
        self.optimizers["a_optimizer"].step()

        for p in self.critic1.parameters():
            p.requires_grad = True
        for p in self.critic2.parameters():
            p.requires_grad = True

        # Update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(
                self.critic1.parameters(), self.critic1_target.parameters()
            ):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
            for p, p_targ in zip(
                self.critic2.parameters(), self.critic2_target.parameters()
            ):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def configure_optimizers(self):
        optimizers = dict()
        actor_params = list(self.actor.parameters())
        critic1_params = list(self.critic1.parameters())
        critic2_params = list(self.critic2.parameters())

        optimizers["a_optimizer"] = torch.optim.Adam(
            actor_params, lr=self.p_lr)
        optimizers["c1_optimizer"] = torch.optim.Adam(
            critic1_params, lr=self.q_lr)
        optimizers["c2_optimizer"] = torch.optim.Adam(
            critic2_params, lr=self.q_lr)
        return optimizers

    def test_agent(self, test_episodes, env, cplexpath, directory):
        epochs = range(test_episodes)  # epoch iterator
        episode_reward = []
        episode_served_demand = []
        episode_rebalancing_cost = []
        for _ in epochs:
            eps_reward = 0
            eps_served_demand = 0
            eps_rebalancing_cost = 0
            obs = env.reset()
            actions = []
            done = False
            while (not done):
                obs, paxreward, done, info, _, _ = env.pax_step(
                    CPLEXPATH=cplexpath, PATH="scenario_nyc4_test", directory=directory)
                eps_reward += paxreward

                o = self.parse_obs(obs, self.device)

                action_rl = self.select_action(
                    o.x, o.edge_index, deterministic=True)
                actions.append(action_rl)

                desiredAcc = {env.region[i]: int(
                    action_rl[i] * dictsum(env.acc, env.time + 1))for i in range(len(env.region))}

                rebAction = solveRebFlow(
                    env, "scenario_nyc4_test", desiredAcc, cplexpath, directory)

                _, rebreward, done, info, _, _ = env.reb_step(rebAction)
                eps_reward += rebreward

                eps_served_demand += info["served_demand"]
                eps_rebalancing_cost += info["rebalancing_cost"]
            episode_reward.append(eps_reward)
            episode_served_demand.append(eps_served_demand)
            episode_rebalancing_cost.append(eps_rebalancing_cost)

        return (
            np.mean(episode_reward),
            np.mean(episode_served_demand),
            np.mean(episode_rebalancing_cost),
        )

    def _get_action_and_values(self, data, num_actions, batch_size, action_dim):
        alpha = torch.ones(action_dim)
        d = torch.distributions.Dirichlet(alpha)
        # sample random actions
        random_actions = d.sample((batch_size*self.num_random,))
        random_log_prob = d.log_prob(random_actions).view(
            batch_size, num_actions, 1).to(self.device)
        random_actions = random_actions.to(self.device)

        data_list = data.to_data_list()
        data_list = data_list*num_actions
        batch_temp = Batch.from_data_list(data_list).to(self.device)

        current_actions, current_log = self.actor(
            batch_temp.x_s, batch_temp.edge_index_s)
        current_log = current_log.view(batch_size, num_actions, 1)

        next_actions, next_log = self.actor(
            batch_temp.x_t, batch_temp.edge_index_t)
        next_log = next_log.view(batch_size, num_actions, 1)

        q1_rand = self.critic1(batch_temp.x_s, batch_temp.edge_index_s, random_actions).view(
            batch_size, num_actions, 1)
        q2_rand = self.critic2(batch_temp.x_s, batch_temp.edge_index_s, random_actions).view(
            batch_size, num_actions, 1)

        q1_current = self.critic1(batch_temp.x_s, batch_temp.edge_index_s, current_actions).view(
            batch_size, num_actions, 1)
        q2_current = self.critic2(batch_temp.x_s, batch_temp.edge_index_s, current_actions).view(
            batch_size, num_actions, 1)

        q1_next = self.critic1(batch_temp.x_s, batch_temp.edge_index_s, next_actions).view(
            batch_size, num_actions, 1)
        q2_next = self.critic2(batch_temp.x_s, batch_temp.edge_index_s, next_actions).view(
            batch_size, num_actions, 1)

        return random_log_prob, current_log, next_log, q1_rand, q2_rand, q1_current, q2_current, q1_next, q2_next

    def save_checkpoint(self, path="ckpt.pth"):
        checkpoint = dict()
        checkpoint["model"] = self.state_dict()
        for key, value in self.optimizers.items():
            checkpoint[key] = value.state_dict()
        torch.save(checkpoint, path)

    def load_checkpoint(self, path="ckpt.pth"):
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint["model"])
        for key, _ in self.optimizers.items():
            self.optimizers[key].load_state_dict(checkpoint[key])

    def log(self, log_dict, path="log.pth"):
        torch.save(log_dict, path)
