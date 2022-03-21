"""
Proximal policy optimization with a few tricks. Adapted from the implementation in baselines.

// Modifications Copyright (c) 2020 Uber Technologies Inc.
"""
import joblib
import numpy as np
import tensorflow as tf
import horovod.tensorflow as hvd
import sys
import copy
import cv2
import logging
logger = logging.getLogger(__name__)


class Runner(object):
    def __init__(self, env, model, nsteps, gamma, lam, norm_adv, subtract_rew_avg):
        self.env = env
        self.model = model
        self.nenv = env.num_envs
        self.gamma = gamma
        self.lam = lam
        self.norm_adv = norm_adv
        self.subtract_rew_avg = subtract_rew_avg
        self.nsteps = nsteps
        # Keep an additional nsteps/2 steps in memory for the purpose of doing back-propagation through time.
        # Seen in: RJ Williams, J Peng,
        # "An efficient gradient-based algorithm for on-line training of recurrent network trajectories" (1990)
        self.num_steps_to_cut_left = nsteps//2
        self.num_steps_to_cut_right = 0
        self.local_traj_counter = 0
        self.steps_taken = 0
        self.to_shift = []

        # Per episode information
        self.epinfos = []
        self.mb_cells = []
        self.mb_game_reward = []
        self.mb_exp_strat = []
        self.mb_traj_index = []
        self.mb_traj_len = []

        self.single_frame_obs_space = env.recursive_getattr('single_frame_obs_space')
        self.mb_goals = self.reg_shift_list()
        self.mb_increase_ent = self.reg_shift_list()
        self.mb_rewards = self.reg_shift_list()
        self.mb_reward_avg = self.reg_shift_list()
        self.mb_actions = self.reg_shift_list()
        self.mb_timesteps = self.reg_shift_list()

        self.mb_valids = self.reg_shift_list()
        self.mb_random_resets = self.reg_shift_list()
        self.mb_dones = self.reg_shift_list()
        self.mb_trajectory_ids = self.reg_shift_list()

        self.ar_mb_goals = None
        self.ar_mb_ent = None
        self.ar_mb_valids = None
        self.ar_mb_actions = None
        self.ar_mb_timesteps = None
        self.ar_mb_dones = None
        self.ar_mb_cells = None
        self.ar_mb_game_reward = None
        self.ar_mb_ret_strat = None
        self.trunc_lst_mb_trajectory_ids = None
        self.trunc_lst_mb_dones = None
        self.trunc_lst_mb_rewards = None
        self.trunc_mb_obs = None
        self.trunc_mb_goals = None
        self.trunc_mb_actions = None
        self.trunc_mb_timesteps = None
        self.trunc_mb_valids = None
        self.ar_mb_traj_index = None
        self.ar_mb_traj_len = None

        self.ar_mb_obs_2 = np.zeros(shape=[self.nenv, self.nsteps + self.num_steps_to_cut_left, 80, 105, 12],
                                    dtype=self.model.train_model.X.dtype.name)
        self.obs_final = None
        self.first_rollout = True

        self.traj_id_limit = None

    def init_obs(self):
        logger.info('Resetting environments...')
        obs_and_goals = self.env.reset()
        obs, goals = obs_and_goals

        logger.info('Casting the observation...')
        self.obs_final = np.cast[self.model.train_model.X.dtype.name](obs)
        logger.info(f'Assigning the observation to a slice of our observation array: {self.obs_final.shape}')
        self.ar_mb_obs_2[:, 0, ...] = self.obs_final

        logger.info('Casting the goal...')
        self.mb_goals.append(np.cast[self.model.train_model.goal.dtype.name](goals))
        logger.info(f'Creating entropy array of size: {self.nenv}')
        self.mb_increase_ent.append(np.ones(self.nenv, dtype=np.float32))
        logger.info(f'Creating random-reset array of size: {self.nenv}')
        self.mb_random_resets.append(np.array([False for _ in range(self.nenv)]))
        logger.info(f'Creating done array of size: {self.nenv}')
        self.mb_dones.append(np.array([False for _ in range(self.nenv)]))
        logger.info(f'Appending initial state of shape: {self.model.initial_state.shape}')
        self.mb_states.append(self.model.initial_state)
        logger.info(f'Creating new trajectory ids of size: {self.nenv}')
        self.mb_trajectory_ids.append(np.array([self.get_next_traj_id() for _ in range(self.nenv)]))
        logger.info('init_obs done!')

    def get_next_traj_id(self):
        result = self.local_traj_counter + hvd.rank() * (sys.maxsize // hvd.size())
        self.local_traj_counter += 1
        return result

    def init_trajectory_id(self, archive):
        relevant = set()
        for trajectory_id in archive.cell_trajectory_manager.cell_trajectories:
            if (hvd.rank()+1) * (sys.maxsize // hvd.size()) > trajectory_id > hvd.rank() * (sys.maxsize // hvd.size()):
                relevant.add(trajectory_id)
        if len(relevant) > 0:
            self.local_traj_counter = (max(relevant) - hvd.rank() * (sys.maxsize // hvd.size())) + 1
        else:
            self.local_traj_counter = 0

    def reg_shift_list(self, initial_value=None):
        if initial_value is not None:
            shifting_list = ShiftingList()
            shifting_list.append(initial_value)
        else:
            shifting_list = ShiftingList()
        self.to_shift.append(shifting_list)
        return shifting_list

    def run(self):
        # shift forward
        if len(self.mb_rewards) >= self.nsteps + self.num_steps_to_cut_left + self.num_steps_to_cut_right:
            for shifting_list in self.to_shift:
                shifting_list.shift(self.nsteps)

        if not self.first_rollout:
            for i in range(self.num_steps_to_cut_left):
                self.ar_mb_obs_2[:, i, ...] = self.ar_mb_obs_2[:, i - self.num_steps_to_cut_left, ...]
            self.ar_mb_obs_2[:, self.num_steps_to_cut_left, ...] = self.obs_final

        # This is information for which we do not need to remember anything from the previous batch
        self.epinfos = []
        self.mb_cells = []
        self.mb_game_reward = []
        self.mb_exp_strat = []
        self.mb_traj_index = []
        self.mb_traj_len = []

        self.steps_taken = 0
        while len(self.mb_rewards) < self.nsteps+self.num_steps_to_cut_left+self.num_steps_to_cut_right:
            self.steps_taken += 1

            actions, values, states, neglogpacs = self.step_model(self.obs_final, self.mb_goals, self.mb_dones, np.zeros((self.nenv)), self.mb_increase_ent)
            obs_and_goals, rewards, dones, infos = self.env.step(actions)
            timesteps = np.zeros(self.nenv) + self.steps_taken
            self.append_mb_data(actions, obs_and_goals, rewards, dones, timesteps, infos)

        # self.mb_timesteps = np.ones(1, self.nenv,1) * self.steps_taken
        # extract arrays
        end = self.nsteps + self.num_steps_to_cut_left
        self.gather_return_info(end)

        self.first_rollout = False

    def get_entropy(self, infos):
        return np.asarray([info.get('increase_entropy', 1.0) for info in infos], dtype=np.float32)

    def step_model(self, obs, mb_goals, mb_dones, timesteps, mb_increase_ent):
        return self.model.step(obs, mb_goals[-1], mb_dones[-1], timesteps, mb_increase_ent[-1])

    def append_mb_data(self, actions, obs_and_goals, rewards, dones, timesteps, infos):
        overwritten_action = [info.get('overwritten_action', -1) for info in infos]
        for i in range(len(actions)):
            if overwritten_action[i] >= 0:
                actions[i] = overwritten_action[i]

        self.mb_actions.append(actions)
        obs, goals = obs_and_goals

        if self.first_rollout:
            write_index = self.steps_taken
        else:
            write_index = self.num_steps_to_cut_left + self.steps_taken
        if write_index < self.ar_mb_obs_2.shape[1]:
            self.ar_mb_obs_2[:, write_index, ...] = np.cast[self.model.train_model.X.dtype.name](obs)
        self.obs_final = np.cast[self.model.train_model.X.dtype.name](obs)

        self.mb_goals.append(np.cast[self.model.train_model.goal.dtype.name](goals))
        self.mb_increase_ent.append(self.get_entropy(infos))
        self.mb_rewards.append(rewards)
        self.mb_timesteps.append(timesteps)

        self.mb_dones.append(dones)
        self.mb_valids.append([(not info.get('replay_reset.invalid_transition', False)) for info in infos])
        self.mb_random_resets.append(np.array([info.get('replay_reset.random_reset', False) for info in infos]))
        self.mb_exp_strat.append(np.array([info.get('exp_strat', 0) for info in infos]))
        self.mb_cells.append([info.get('cell') for info in infos])
        self.mb_game_reward.append([info.get('game_reward') for info in infos])
        self.mb_traj_index.append([info.get('traj_index', -1) for info in infos])
        self.mb_traj_len.append([info.get('traj_len', -1) for info in infos])
        traj_ids = copy.copy(self.mb_trajectory_ids[-1])
        for i, done in enumerate(dones):
            if done:
                traj_ids[i] = self.get_next_traj_id()
        self.mb_trajectory_ids.append(traj_ids)

        for i, info in enumerate(infos):
            maybeepinfo = info.get('episode')
            if maybeepinfo:
                if self.traj_id_limit is not None:
                    if self.mb_trajectory_ids[-2][i] >= self.traj_id_limit:
                        continue
                self.epinfos.append(maybeepinfo)

    def gather_return_info(self, end):
        from baselines.common.mpi_moments import mpi_moments
        self.ar_mb_goals = sf01(np.asarray(self.mb_goals[:end], dtype=self.model.train_model.goal.dtype.name), 'goals')
        self.ar_mb_ent = sf01(np.stack(self.mb_increase_ent[:end], axis=0), 'ents')
        self.ar_mb_valids = sf01(np.asarray(self.mb_valids[:end], dtype=np.float32), 'valids')
        self.ar_mb_actions = sf01(np.asarray(self.mb_actions[:end]), 'actions')
        self.ar_mb_timesteps = sf01(np.asarray(self.mb_timesteps[:end]), 'timesteps')
        self.ar_mb_dones = sf01(np.asarray(self.mb_dones[:end], dtype=np.bool), 'dones')

        self.ar_mb_cells = sf01(np.asarray(self.mb_cells, dtype=np.object), 'cells')
        self.ar_mb_ret_strat = sf01(np.asarray(self.mb_exp_strat, dtype=np.int32), 'ret_strats')

        self.ar_mb_game_reward = sf01(np.asarray(self.mb_game_reward, dtype=np.float32), 'game_rew')

        trunc_trajectory_ids = self.mb_trajectory_ids[-len(self.mb_cells) - 1:len(self.mb_trajectory_ids) - 1]
        self.trunc_lst_mb_trajectory_ids = sf01(np.asarray(trunc_trajectory_ids, dtype=np.int), 'traj_ids')
        trunc_dones = self.mb_dones[-len(self.mb_cells):len(self.mb_dones)]
        self.trunc_lst_mb_dones = sf01(np.asarray(trunc_dones, dtype=np.bool), 'trunc_dones')
        trunc_rewards = self.mb_rewards[-len(self.mb_cells):len(self.mb_rewards)]
        self.trunc_lst_mb_rewards = sf01(np.asarray(trunc_rewards, dtype=np.float32), 'trunc_rews')

        single_frames = []
        nb_channels = self.single_frame_obs_space.shape[-1]
        stacks_times_nb_channels = self.ar_mb_obs_2[0].shape[-1]
        last_frame_start = stacks_times_nb_channels - nb_channels
        if self.first_rollout:
            start = 0
        else:
            start = self.num_steps_to_cut_left
        for env_i in range(self.ar_mb_obs_2.shape[0]):
            for it_i in range(start, self.ar_mb_obs_2.shape[1]):
                frame = self.ar_mb_obs_2[env_i, it_i, :, :, last_frame_start:]
                single_frames.append(cv2.imencode('.png', frame, [cv2.IMWRITE_PNG_COMPRESSION, 7])[1])

        self.trunc_mb_obs = single_frames
        self.trunc_mb_goals = sf01(np.asarray(self.mb_goals[end-len(self.mb_cells):end],
                                              dtype=self.model.train_model.goal.dtype.name), 'trunc_goals')
        self.trunc_mb_actions = sf01(np.asarray(self.mb_actions[end-len(self.mb_cells):end],
                                                dtype=np.int), 'trunc_acts')
        self.trunc_timesteps = sf01(np.asarray(self.mb_timesteps[end-len(self.mb_cells):end],
                                                dtype=np.int), 'trunc_steps')
        self.trunc_mb_valids = sf01(np.asarray(self.mb_valids[end-len(self.mb_cells):end],
                                               dtype=np.int), 'trunc_valids')

        self.ar_mb_traj_index = sf01(np.asarray(self.mb_traj_index, dtype=np.int32), 'ar_mb_traj_index')
        self.ar_mb_traj_len = sf01(np.asarray(self.mb_traj_len, dtype=np.int32), 'ar_mb_traj_len')


def sf01(arr, _name=''):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    new_array = arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])
    return new_array


def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)


def flatten_lists(listoflists):
    """
    Turns a list of [[a,b], [c, d]] into [a, b, c, d]

    @param listoflists:
    @return:
    """
    return [el for list_ in listoflists for el in list_]
