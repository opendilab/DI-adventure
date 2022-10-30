"""
智能体评估函数
"""
import torch
from ding.utils import set_pkg_seed
from baseline_task_pipeline import main_config, wrapped_mario_env, create_config
from model import DQN
from policy import DQNPolicy
from ding.config import compile_config


def evaluate(state_dict, seed, video_dir_path, eval_times):
    cfg = compile_config(main_config, create_cfg=create_config, auto=True)
    env = wrapped_mario_env()
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
    ckpt_path = './eval.pth.tar'
    state_dict = torch.load(ckpt_path, map_location='cpu')
    evaluate(state_dict=state_dict, seed=0, video_dir_path='eval_videos', eval_times=4)
