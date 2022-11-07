from typing import TYPE_CHECKING, Callable
import numpy as np
from ding.utils import DistributedWriter
if TYPE_CHECKING:
    from ding.framework import OnlineRLContext


def get_hist_gif(data, max_length, action_shape):
    import os
    import cv2
    from matplotlib import pyplot as plt
    from matplotlib import animation

    def animate(num, dist, rects):
        for rect, x in zip(rects, dist[num]):
            rect.set_height(x)
        return rects

    path = 'tmp.gif'

    new_data = []
    for q_value_dist in data:
        fig = plt.figure()
        rects = plt.bar(
            [i for i in range(action_shape)],
            q_value_dist[0],
            color=["orange", "yellow", "red", "blue", "purple", "green", "darkcyan"]
        )
        anim = animation.FuncAnimation(fig, animate, fargs=(q_value_dist, rects), frames=max_length, blit=True)
        anim.save(path, writer='pillow')
        plt.clf()
        handle = cv2.VideoCapture(path)
        q_value_dist = []
        while True:
            ret, frame = handle.read()
            if not ret:
                break
            q_value_dist.append(frame.transpose(2, 0, 1))
        new_data.append(np.stack(q_value_dist))
    new_data = np.stack(new_data)

    os.remove('tmp.gif')
    return new_data


def online_logger(record_train_iter: bool = False, train_show_freq: int = 100, video_save_freq: int = int(5e5)) -> Callable:
    writer = DistributedWriter.get_instance()
    last_train_show_iter = -1
    last_video_save_step = -1

    def _logger(ctx: "OnlineRLContext"):
        nonlocal last_train_show_iter
        nonlocal last_video_save_step
        if not np.isinf(ctx.eval_value):
            if record_train_iter:
                writer.add_scalar('basic/eval_episode_return_mean-env_step', ctx.eval_value, ctx.env_step)
                writer.add_scalar('basic/eval_episode_return_mean-train_iter', ctx.eval_value, ctx.train_iter)
                writer.add_scalar(
                    'basic/eval_episode_discount_return_mean-env_step',
                    ctx.eval_output['episode_info']['discount_return_mean'], ctx.env_step
                )
                writer.add_scalar(
                    'basic/eval_episode_discount_return_mean-train_iter',
                    ctx.eval_output['episode_info']['discount_return_mean'], ctx.train_iter
                )
                writer.add_scalar('basic/exploration_epsilon-env_step', ctx.collect_kwargs['eps'], ctx.env_step)
                writer.add_scalar('basic/exploration_epsilon-train_iter', ctx.collect_kwargs['eps'], ctx.train_iter)
            else:
                writer.add_scalar('basic/eval_episode_return_mean', ctx.eval_value, ctx.env_step)
                writer.add_scalar(
                    'basic/eval_episode_discount_return_mean', ctx.eval_output['episode_info']['discount_return_mean'],
                    ctx.env_step
                )
                writer.add_scalar('basic/exploration_epsilon', ctx.collect_kwargs['eps'], ctx.env_step)
            if (ctx.env_step - last_video_save_step) > video_save_freq:
                # save replay video
                writer.add_video('eval_replay_videos', ctx.eval_output['replay_video'], ctx.env_step, 30)
                output = ctx.eval_output['output']
                # save q distribution
                q_value_dist = [np.stack([t['logit'] for t in o]) for o in output]
                max_length = max(q.shape[0] for q in q_value_dist)
                action_shape = q_value_dist[0].shape[1]
                for i in range(len(q_value_dist)):
                    N = q_value_dist[i].shape[0]
                    if N < max_length:
                        q_value_dist[i] = np.concatenate(
                            [q_value_dist[i]] + [q_value_dist[:-1] for _ in range(max_length - N)]
                        )
                q_value_dist = get_hist_gif(q_value_dist, max_length, action_shape)
                writer.add_video('eval_q_value_distribution', q_value_dist, ctx.env_step, 30)
                writer.flush()
                last_video_save_step = ctx.env_step
        if ctx.train_output is not None and ctx.train_iter - last_train_show_iter >= train_show_freq:
            last_train_show_iter = ctx.train_iter
            output = ctx.train_output.pop()
            for k, v in output.items():
                # directly output from policy._forward_learn
                if record_train_iter:
                    writer.add_scalar('basic/train_{}-train_iter'.format(k), v, ctx.train_iter)
                    writer.add_scalar('basic/train_{}-env_step'.format(k), v, ctx.env_step)
                else:
                    writer.add_scalar('basic/train_{}'.format(k), v, ctx.env_step)
    return _logger
