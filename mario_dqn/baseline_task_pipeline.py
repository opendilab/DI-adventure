from easydict import EasyDict
from ditk import logging
from ding.envs import DingEnvWrapper, SubprocessEnvManagerV2
from ding.data import DequeBuffer
from ding.config import compile_config
from ding.framework import task, ding_init
from ding.framework.context import OnlineRLContext
from ding.framework.middleware import OffPolicyLearner, StepCollector, interaction_evaluator, data_pusher, \
    eps_greedy_handler, CkptSaver, nstep_reward_enhancer, termination_checker
from ding.utils import set_pkg_seed
# env import
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
# algo import
from policy import DQNPolicy
from model import DQN
from wrapper import MaxAndSkipWrapper, WarpFrameWrapper, ScaledFloatFrameWrapper, FinalEvalRewardEnv, \
    FrameStackWrapper
from middleware import online_logger

mario_dqn_config = dict(
    exp_name='exp/mario_dqn_baseline',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=100000,
        # replay_path='mario_dqn_baseline/video',
    ),
    policy=dict(
        cuda=True,
        model=dict(
            obs_shape=[1, 84, 84],
            action_shape=7,
            encoder_hidden_size_list=[32, 64, 128],
        ),
        nstep=3,
        discount_factor=0.99,
        learn=dict(
            update_per_collect=10,
            batch_size=32,
            learning_rate=0.0001,
        ),
    ),
)
mario_dqn_config = EasyDict(mario_dqn_config)
main_config = mario_dqn_config
mario_dqn_create_config = dict(
    env_manager=dict(type='subprocess'),
    policy=dict(type='mario_dqn'),
)
mario_dqn_create_config = EasyDict(mario_dqn_create_config)
create_config = mario_dqn_create_config


def wrapped_mario_env():
    return DingEnvWrapper(
        JoypadSpace(gym_super_mario_bros.make("SuperMarioBros-1-1-v0"), SIMPLE_MOVEMENT),
        cfg={
            'env_wrapper': [
                lambda env: MaxAndSkipWrapper(env, skip=4),
                lambda env: WarpFrameWrapper(env, size=84),
                lambda env: ScaledFloatFrameWrapper(env),
                lambda env: FrameStackWrapper(env, n_frames=1),
                lambda env: FinalEvalRewardEnv(env),
            ]
        }
    )


def main(mario_main_config, mario_create_config, seed):
    cfg = compile_config(cfg=mario_main_config, create_cfg=mario_create_config, auto=True, seed=seed)
    filename = '{}/log.txt'.format(cfg.exp_name)
    logging.getLogger(with_files=[filename]).setLevel(logging.INFO)
    ding_init(cfg)
    with task.start(ctx=OnlineRLContext()):
        collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num
        collector_env = SubprocessEnvManagerV2(
            env_fn=[wrapped_mario_env for _ in range(collector_env_num)], cfg=cfg.env.manager
        )
        evaluator_env = SubprocessEnvManagerV2(
            env_fn=[wrapped_mario_env for _ in range(evaluator_env_num)], cfg=cfg.env.manager
        )

        set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)

        model = DQN(**cfg.policy.model)
        buffer_ = DequeBuffer(size=cfg.policy.other.replay_buffer.replay_buffer_size)
        policy = DQNPolicy(cfg.policy, model=model)

        task.use(interaction_evaluator(cfg, policy.eval_mode, evaluator_env))
        task.use(eps_greedy_handler(cfg))
        task.use(StepCollector(cfg, policy.collect_mode, collector_env))
        task.use(nstep_reward_enhancer(cfg))
        task.use(data_pusher(cfg, buffer_))
        task.use(OffPolicyLearner(cfg, policy.learn_mode, buffer_))
        task.use(online_logger(record_train_iter=False, train_show_freq=500))
        task.use(CkptSaver(cfg, policy, train_freq=10000))
        task.use(termination_checker(max_env_step=int(1e7)))
        task.run()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", "-s", type=int, default=0)
    args = parser.parse_args()
    main_config.exp_name += "_seed"+str(args.seed)
    main(main_config, create_config, seed=args.seed)
