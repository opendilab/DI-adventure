from typing import TYPE_CHECKING, Callable
import numpy as np
from ding.utils import DistributedWriter
if TYPE_CHECKING:
    from ding.framework import OnlineRLContext, OfflineRLContext


def online_logger(record_train_iter: bool = False, train_show_freq: int = 100) -> Callable:
    writer = DistributedWriter.get_instance()
    last_train_show_iter = -1

    def _logger(ctx: "OnlineRLContext"):
        nonlocal last_train_show_iter
        if not np.isinf(ctx.eval_value):
            if record_train_iter:
                writer.add_scalar('basic/eval_episode_reward_mean-env_step', ctx.eval_value, ctx.env_step)
                writer.add_scalar('basic/eval_episode_reward_mean-train_iter', ctx.eval_value, ctx.train_iter)
                writer.add_scalar('basic/exploration_epsilon-env_step', ctx.collect_kwargs['eps'], ctx.env_step)
                writer.add_scalar('basic/exploration_epsilon-train_iter', ctx.collect_kwargs['eps'], ctx.train_iter)
            else:
                writer.add_scalar('basic/eval_episode_reward_mean', ctx.eval_value, ctx.env_step)
                writer.add_scalar('basic/exploration_epsilon', ctx.collect_kwargs['eps'], ctx.env_step)
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
