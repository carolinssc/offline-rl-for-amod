from __future__ import print_function
import argparse
from tqdm import trange
import numpy as np
import torch

from envs.amod_env import Scenario, AMoD
from algos.c_sac import SAC
from src.algos.reb_flow_solver import solveRebFlow
from src.misc.utils import dictsum
import random
import pickle
from torch_geometric.data import Data, Batch
import json
import OmegaConf


class PairData(Data):
    """
    Store 2 graphs in one Data object (s_t and s_t+1)
    """

    def __init__(self, edge_index_s=None, x_s=None, reward=None, action=None, edge_index_t=None, x_t=None):
        super().__init__()
        self.edge_index_s = edge_index_s
        self.x_s = x_s
        self.reward = reward
        self.action = action
        self.edge_index_t = edge_index_t
        self.x_t = x_t

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)


class ReplayData:
    """
    Replay buffer for SAC agents
    """

    def __init__(self, device, rew_scale):
        self.device = device
        self.data_list = []
        self.rew_scale = rew_scale

    def create_dataset(self, edge_index, memory_path, size=60000, st=False, sc=False):
        w = open(f'/replaymemories/{memory_path}.pkl', "rb")

        replay_buffer = pickle.load(w)
        data = replay_buffer.sample_all(size)
        if st:
            mean = data['rew'].mean()
            std = data['rew'].std()
            data['rew'] = (data['rew']-mean)/(std + 1e-16)
        elif sc:
            data['rew'] = (data['rew'] - data['rew'].min()) / \
                (data['rew'].max() - data['rew'].min())

        print(data['rew'].min())
        print(data['rew'].max())

        (state_batch, action_batch, reward_batch, next_state_batch) = (
            data["obs"], data["act"], args.rew_scale*data["rew"], data["obs2"])
        for i in range(len(state_batch)):
            self.data_list.append(PairData(
                edge_index, state_batch[i], reward_batch[i], action_batch[i], edge_index, next_state_batch[i]))

    def sample_batch(self, batch_size=32):
        data = random.sample(self.data_list, batch_size)
        return Batch.from_data_list(data, follow_batch=['x_s', 'x_t']).to(self.device)


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


parser = argparse.ArgumentParser(description='SAC-GNN')

demand_ratio = {'san_francisco': 2, 'washington_dc': 4.2, 'nyc_brooklyn': 9, 'rome': 1.8,
                'shenzhen_downtown_west': 2.5}
json_hr = {'san_francisco': 19, 'washington_dc': 19, 'nyc_brooklyn': 19, 'rome': 8,
           'shenzhen_downtown_west': 8}
beta = {'san_francisco': 0.2, 'washington_dc': 0.5, 'nyc_brooklyn': 0.5, 'porto': 0.1, 'rome': 0.1,
        'shenzhen_downtown_west': 0.5}

test_tstep = {'san_francisco': 3,
              'nyc_brooklyn': 4, 'shenzhen_downtown_west': 3}

# Simulator parameters
parser.add_argument('--seed', type=int, default=10, metavar='S',
                    help='random seed (default: 10)')
parser.add_argument('--demand_ratio', type=int, default=0.5, metavar='S',
                    help='demand_ratio (default: 0.5)')
parser.add_argument('--json_hr', type=int, default=7, metavar='S',
                    help='json_hr (default: 7)')
parser.add_argument('--json_tstep', type=int, default=3, metavar='S',
                    help='minutes per timestep (default: 3min)')
parser.add_argument('--beta', type=int, default=0.5, metavar='S',
                    help='cost of rebalancing (default: 0.5)')

# Model parameters
parser.add_argument('--test', type=bool, default=False,
                    help='activates test mode for agent evaluation')
parser.add_argument('--cplexpath', type=str, default="/opt/opl/bin/x86-64_linux/",
                    help='defines directory of the CPLEX installation')
parser.add_argument('--directory', type=str, default='saved_files',
                    help='defines directory where to save files')
parser.add_argument('--max_episodes', type=int, default=10000, metavar='N',
                    help='number of episodes to train agent (default: 16k)')
parser.add_argument('--max_steps', type=int, default=20, metavar='N',
                    help='number of steps per episode (default: T=60)')
parser.add_argument('--cuda', type=bool, default=True,
                    help='enables CUDA training')
parser.add_argument("--batch_size", type=int, default=100,
                    help='defines the batch size')
parser.add_argument("--alpha", type=float, default=0.3,
                    help='value of the entropy coefficient')
parser.add_argument("--hidden_size", type=int, default=256,
                    help='number of hidden units in the MLP layer')
parser.add_argument("--checkpoint_path", type=str, default='SAC',
                    help='path, where to save model checkpoints')
parser.add_argument("--city", type=str, default='shenzhen_downtown_west',
                    help='defines city to train on')

# CQL parameters
parser.add_argument("--load_yaml", type=bool, default=False,
                    help='to load CQL parameters from a yaml file')
parser.add_argument("--memory_path", type=str, default='SAC',
                    help='path, where data is saved')
parser.add_argument("--min_q_weight", type=float, default=5,
                    help='conservatie coeffiecent (eta in paper)')
parser.add_argument("--samples_buffer", type=int, default=10000,
                    help='number of samples to take from the dataset')
parser.add_argument("--lagrange_thresh", type=float, default=-1,
                    help='lagrange treshhold tau for automatic tuning of eta')
parser.add_argument("--rew_scale", type=float, default=0.1,
                    help='defines reward scale')
parser.add_argument("--st", type=bool, default=False,
                    help='whether to standardize data')
parser.add_argument("--sc", type=bool, default=False,
                    help='wether to scale data')
args = parser.parse_args()
args.cuda = args.cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
city = args.city
# Define AMoD Simulator Environment

if not args.test:
    if args.load_yaml == True:
        parameter = OmegaConf.load(f"src/conf/config_{city}.yaml")
        args.memory = parameter.memory_path
        args.min_q_weight = parameter.min_q_weight
        args.samples_buffer = parameter.samples_buffer
        args.lagrange_thresh = parameter.lagrange_thresh
        args.rew_scale = parameter.rew_scale
        args.max_episodes = parameter.max_episodes

    scenario = Scenario(json_file=f"data/scenario_{city}.json", demand_ratio=demand_ratio[city],
                        json_hr=json_hr[city], sd=args.seed, json_tstep=args.json_tstep, tf=args.max_steps)

    env = AMoD(scenario, beta=beta[city])

    model = SAC(
        env=env,
        input_size=13,
        hidden_size=args.hidden_size,
        p_lr=1e-4,
        q_lr=3e-4,
        alpha=args.alpha,
        batch_size=args.batch_size,
        deterministic_backup=True,
        min_q_weight=args.min_q_weight,
        use_automatic_entropy_tuning=False,
        lagrange_thresh=args.lagrange_thresh,
        n=args.n,
        device=device,
        json_file=f'data/scenario_{city}.json',
        min_q_version=3,
    ).to(device)

    with open(f'data/scenario_{city}.json', "r") as file:
        data = json.load(file)

    edge_index = torch.vstack((torch.tensor([edge['i'] for edge in data["topology_graph"]]).view(1, -1),
                               torch.tensor([edge['j'] for edge in data["topology_graph"]]).view(1, -1))).long()
    # Initialize Dataset
    Dataset = ReplayData(device=device, rew_scale=args.rew_scale)
    Dataset.create_dataset(edge_index=edge_index, memory_path=args.memory_path,
                           size=args.samples_buffer, st=args.st, sc=args.sc)
    # Initialize lists for logging
    log = {'train_reward': [],
           'train_served_demand': [],
           'train_reb_cost': []}
    train_episodes = args.max_episodes  # set max number of training episodes
    T = args.max_steps  # set episode length
    epochs = trange(train_episodes)  # epoch iterator
    best_reward = -np.inf  # set best reward
    model.train()  # set model in train mode

    # Training Loop
    for step in range(train_episodes*20):
        if step % 400 == 0:
            episode_reward, episode_served_demand, episode_rebalancing_cost = model.test_agent(
                2, env, args.cplexpath, args.directory)

            epochs.set_description(
                f"Episode {step/20} | Reward: {episode_reward:.2f} | ServedDemand: {episode_served_demand:.2f} | Reb. Cost: {episode_rebalancing_cost:.2f}")

        # Checkpoint best performing model
            if episode_reward >= best_reward and step > 1000:
                model.save_checkpoint(
                    path=f"ckpt/{args.checkpoint_path}.pth")
                best_reward = episode_reward

        batch = Dataset.sample_batch(args.batch_size)
        model.update(data=batch, conservative=True)

else:
    # test pre-trained model
    if city == 'nyc_brooklyn':
        scenario = Scenario(json_file=f"data/scenario_{city}.json", demand_ratio=demand_ratio[city],
                            json_hr=json_hr[city], sd=args.seed, json_tstep=test_tstep[city], tf=args.max_steps)
    else:
        scenario = Scenario(json_file=f"data/scenario_{city}.json", demand_ratio=demand_ratio[city],
                            json_hr=json_hr[city], sd=args.seed, json_tstep=args.json_tstep, tf=args.max_steps)

    env = AMoD(scenario, beta=beta[city])

    model = SAC(
        env=env,
        input_size=13,
        hidden_size=args.hidden_size,
        p_lr=1e-4,
        q_lr=3e-4,
        alpha=args.alpha,
        batch_size=args.batch_size,
        deterministic_backup=True,
        min_q_weight=args.min_q_weight,
        use_automatic_entropy_tuning=False,
        lagrange_thresh=args.lagrange_thresh,
        n=args.n,
        device=device,
        json_file=f'data/scenario_{city}.json',
    ).to(device)

    # Load pre-trained model
    model.load_checkpoint(path=f"/ckpt/{args.checkpoint_path}.pth")

    test_episodes = args.max_episodes  # set max number of training episodes
    T = args.max_steps  # set episode length
    epochs = trange(test_episodes)  # epoch iterator
    # Initialize lists for logging
    log = {'test_reward': [],
           'test_served_demand': [],
           'test_reb_cost': []}

    for episode in range(10):
        episode_reward = 0
        episode_served_demand = 0
        episode_rebalancing_cost = 0
        obs = env.reset()
        done = False
        while (not done):
            # take matching step (Step 1 in paper)
            obs, paxreward, done, info, _, _ = env.pax_step(
                CPLEXPATH=args.cplexpath, PATH="scenario_nyc4_test", directory=args.directory)

            episode_reward += paxreward
            # use GNN-RL policy (Step 2 in paper)
            o = model.parse_obs(obs, device=device)
            action_rl = model.select_action(
                o.x, o.edge_index, deterministic=True)
            # transform sample from Dirichlet into actual vehicle counts (i.e. (x1*x2*..*xn)*num_vehicles)
            desiredAcc = {env.region[i]: int(
                action_rl[i] * dictsum(env.acc, env.time+1))for i in range(len(env.region))}
            # solve minimum rebalancing distance problem (Step 3 in paper)
            rebAction = solveRebFlow(
                env, "scenario_nyc4_test", desiredAcc, args.cplexpath, args.directory)

            _, rebreward, done, info, _, _ = env.reb_step(rebAction)

            episode_reward += rebreward
            # track performance over episode
            episode_served_demand += info['served_demand']
            episode_rebalancing_cost += info['rebalancing_cost']
        # Send current statistics to screen
        epochs.set_description(
            f"Episode {episode+1} | Reward: {episode_reward:.2f} | ServedDemand: {episode_served_demand:.2f} | Reb. Cost: {episode_rebalancing_cost}")
        # Log KPIs
