import os
import gym
from tensorboardX import SummaryWriter
from easydict import EasyDict

from ding.config import compile_config
from ding.worker import BaseLearner, SampleSerialCollector, InteractionSerialEvaluator, AdvancedReplayBuffer
from ding.envs import SyncSubprocessEnvManager, DingEnvWrapper, BaseEnvManager
from wrapper import MaxAndSkipWrapper, WarpFrameWrapper, ScaledFloatFrameWrapper, FrameStackWrapper, \
    FinalEvalRewardEnv
from policy import DQNPolicy
from model import DQN
from ding.utils import set_pkg_seed
from ding.rl_utils import get_epsilon_greedy_fn
from mario_dqn_config import mario_dqn_config
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
from functools import partial


action_dict = {2: [["right"], ["right", "A"]], 7: SIMPLE_MOVEMENT, 12: COMPLEX_MOVEMENT}
action_nums = [2, 7, 12]


def wrapped_mario_env(version=0, action=7, obs=1):
    return DingEnvWrapper(
        JoypadSpace(gym_super_mario_bros.make("SuperMarioBros-1-1-v"+str(version)), action_dict[int(action)]),
        cfg={
            'env_wrapper': [
                lambda env: MaxAndSkipWrapper(env, skip=4),
                lambda env: WarpFrameWrapper(env, size=84),
                lambda env: ScaledFloatFrameWrapper(env),
                lambda env: FrameStackWrapper(env, n_frames=obs),
                lambda env: FinalEvalRewardEnv(env),
            ]
        }
    )


def main(cfg, args, seed=0, max_env_step=int(1e7)):
    cfg = compile_config(
        cfg,
        SyncSubprocessEnvManager,
        DQNPolicy,
        BaseLearner,
        SampleSerialCollector,
        InteractionSerialEvaluator,
        AdvancedReplayBuffer,
        seed=seed,
        save_cfg=True
    )
    collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num
    collector_env = SyncSubprocessEnvManager(
        env_fn=[partial(wrapped_mario_env, version=args.version, action=args.action, obs=args.obs) for _ in range(collector_env_num)], cfg=cfg.env.manager
    )
    evaluator_env = SyncSubprocessEnvManager(
        env_fn=[partial(wrapped_mario_env, version=args.version, action=args.action, obs=args.obs) for _ in range(evaluator_env_num)], cfg=cfg.env.manager
    )

    # Set random seed for all package and instance
    collector_env.seed(seed)
    evaluator_env.seed(seed, dynamic_seed=False)
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)

    # Set up RL Policy
    model = DQN(**cfg.policy.model)
    policy = DQNPolicy(cfg.policy, model=model)

    # Set up collection, training and evaluation utilities
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    collector = SampleSerialCollector(
        cfg.policy.collect.collector, collector_env, policy.collect_mode, tb_logger, exp_name=cfg.exp_name
    )
    evaluator = InteractionSerialEvaluator(
        cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    )
    replay_buffer = AdvancedReplayBuffer(cfg.policy.other.replay_buffer, tb_logger, exp_name=cfg.exp_name)

    # Set up other modules, etc. epsilon greedy
    eps_cfg = cfg.policy.other.eps
    epsilon_greedy = get_epsilon_greedy_fn(eps_cfg.start, eps_cfg.end, eps_cfg.decay, eps_cfg.type)

    # Training & Evaluation loop
    while True:
        # Evaluating at the beginning and with specific frequency
        if evaluator.should_eval(learner.train_iter):
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop:
                break
        # Update other modules
        eps = epsilon_greedy(collector.envstep)
        # Sampling data from environments
        new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs={'eps': eps})
        replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
        # Training
        for i in range(cfg.policy.learn.update_per_collect):
            train_data = replay_buffer.sample(learner.policy.get_attribute('batch_size'), learner.train_iter)
            if train_data is None:
                break
            learner.train(train_data, collector.envstep)
        if collector.envstep >= max_env_step:
            break
    # evaluate
    evaluator_env = BaseEnvManager(
        env_fn=[wrapped_mario_env for _ in range(evaluator_env_num)], cfg=cfg.env.manager
    )
    evaluator_env.enable_save_replay(cfg.env.replay_path)  # switch save replay interface
    evaluator = InteractionSerialEvaluator(
        cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    )
    evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)


if __name__ == "__main__":
    from copy import deepcopy
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", "-s", type=int, default=0)
    parser.add_argument("--version", "-v", type=int, default=0, choices=[0,1,2,3])
    parser.add_argument("--action", "-a", type=int, default=7, choices=[2,7,14])
    parser.add_argument("--obs", "-o", type=int, default=1, choices=[1,4])
    args = parser.parse_args()
    mario_dqn_config.exp_name = 'exp/v'+str(args.version)+'_'+str(args.action)+'a_'+str(args.obs)+'f_seed'+str(args.seed)
    mario_dqn_config.policy.model.obs_shape=[args.obs, 84, 84]
    mario_dqn_config.policy.model.action_shape=args.action
    main(deepcopy(mario_dqn_config), args, seed=args.seed)