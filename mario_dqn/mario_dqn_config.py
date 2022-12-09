"""
config 配置文件，这一部分主要包含一些超参数的配置，大家只用关注 model 中的参数即可
"""
from easydict import EasyDict

mario_dqn_config = dict(
    # 实验结果的存放路径
    exp_name='exp/mario_dqn_seed0',
    # mario环境相关
    env=dict(
        # 用来收集经验（experience）的mario环境的数目
        # 请根据机器的性能自行增减
        collector_env_num=8,
        # 用来评估智能体性能的mario环境的数目
        # 请根据机器的性能自行增减
        evaluator_env_num=8,
        # 评估轮次
        n_evaluator_episode=8,
        # 训练停止的分数（3000分可以认为通关1-1，停止训练以节省计算资源）
        stop_value=3000
    ),
    policy=dict(
        # 是否使用 CUDA 加速（必要）
        cuda=True,
        # 神经网络模型相关参数
        model=dict(
            # 网络输入的张量形状
            obs_shape=[1, 84, 84],
            # 有多少个可选动作
            action_shape=7,
            # 网络结构超参数
            encoder_hidden_size_list=[32, 64, 128],
            # 是否使用对决网络 Dueling Network
            dueling=False,
        ),
        # n-step TD
        nstep=3,
        # 折扣系数 gamma
        discount_factor=0.99,
        # 训练相关参数
        learn=dict(
            # 每次利用相同的经验更新网络的次数
            update_per_collect=10,
            # batch size大小
            batch_size=32,
            # 学习率
            learning_rate=0.0001,
            # target Q-network更新频率
            target_update_freq=500,
        ),
        # 收集经验相关，每次收集96个transition进行一次训练
        collect=dict(n_sample=96, ),
        # 评估相关，每2000个iteration评估一次
        eval=dict(evaluator=dict(eval_freq=2000, )),
        other=dict(
            # epsilon-greedy算法
            eps=dict(
                type='exp',
                start=1.,
                end=0.05,
                decay=250000,
            ),
            # replay buffer大小
            replay_buffer=dict(replay_buffer_size=100000, ),
        ),
    ),
)
mario_dqn_config = EasyDict(mario_dqn_config)
main_config = mario_dqn_config
mario_dqn_create_config = dict(
    env_manager=dict(type='subprocess'),
    policy=dict(type='dqn'),
)
mario_dqn_create_config = EasyDict(mario_dqn_create_config)
create_config = mario_dqn_create_config
# you can run `python3 -u mario_dqn_main.py`