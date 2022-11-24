"""
智能体评估函数
"""
import torch
from ding.utils import set_pkg_seed
from mario_dqn_config import mario_dqn_config, mario_dqn_create_config
from model import DQN
from policy import DQNPolicy
from ding.config import compile_config
from ding.envs import DingEnvWrapper
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
from wrapper import MaxAndSkipWrapper, WarpFrameWrapper, ScaledFloatFrameWrapper, FrameStackWrapper, \
    FinalEvalRewardEnv

action_dict = {2: [["right"], ["right", "A"]], 7: SIMPLE_MOVEMENT, 12: COMPLEX_MOVEMENT}
action_nums = [2, 7, 12]


def wrapped_mario_env(version=0, action=2, obs=1):
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


def evaluate(args, state_dict, seed, video_dir_path, eval_times):
    cfg = compile_config(mario_dqn_config, create_cfg=mario_dqn_create_config, auto=True, save_cfg=False)
    env = wrapped_mario_env(args.version, args.action, args.obs)
    model = DQN(**cfg.policy.model)
    model.load_state_dict(state_dict['model'])
    policy = DQNPolicy(cfg.policy, model=model).eval_mode
    env.seed(seed)
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)
    env.enable_save_replay(video_dir_path)
    eval_reward_list = []
    for n in range(eval_times):
        obs = env.reset()
        eval_reward = 0
        while True:
            Q = policy.forward({0: obs})
            action = Q[0]['action'].item()
            obs, reward, done, info = env.step(action)
            eval_reward += reward
            if done or info['time'] < 250:
                print(info)
                eval_reward_list.append(eval_reward)
                break
        print('During {}th evaluation, the total reward your mario got is {}'.format(n, eval_reward))
    print('Eval is over! The performance of your RL policy is {}'.format(sum(eval_reward_list) / len(eval_reward_list)))
    print("Your mario video is saved in {}".format(video_dir_path))
    try:
        del env
    except Exception:
        pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", "-s", type=int, default=0)
    parser.add_argument("--checkpoint", "-ckpt", type=str, default='./exp/mario_dqn_baseline/ckpt/eval.pth.tar')
    parser.add_argument("--replay_path", "-rp", type=str, default='./eval_videos')
    parser.add_argument("--version", "-v", type=int, default=0, choices=[0,1,2,3])
    parser.add_argument("--action", "-a", type=int, default=2, choices=[2,7,14])
    parser.add_argument("--obs", "-o", type=int, default=1, choices=[1,4])
    args = parser.parse_args()
    mario_dqn_config.policy.model.obs_shape=[args.obs, 84, 84]
    mario_dqn_config.policy.model.action_shape=args.action
    ckpt_path = args.checkpoint
    video_dir_path = args.replay_path
    state_dict = torch.load(ckpt_path, map_location='cpu')
    evaluate(args, state_dict=state_dict, seed=args.seed, video_dir_path=video_dir_path, eval_times=1)
