# 强化学习大作业代码配置与运行
> 同学们不要对于RL背后的数学原理和复杂的代码逻辑感到困扰，首先是本次大作业会很少涉及到这一部分，仓库中对这一部分都有着良好的封装；其次是有问题（原理或者代码实现上的）可以随时提问一起交流，方式包括但不限于：
> - github issue
> - 课程微信群
> - 开发者邮箱: opendilab@pjlab.org.cn

## 1. Baseline 代码获取与环境安装
> mario环境的安装教程在网络学堂上，可以自行取用，需要注意：
> 1. 请选择3.8版本的python以避免不必要的版本问题；
> 2. 通过键盘与环境交互需要有可以用于渲染的显示设备，大部分服务器不能胜任，可以选择本地设备或者暂时跳过这一步，对后面没有影响；
### 深度学习框架 PyTorch 安装
这一步有网上有非常多的教程，根据自己的情况安装即可，这里给出一个示例
> 请安装 1.10.0 版本以避免不必要的环境问题
```bash
# 确保您当前是conda环境，且有适合的 GPU 以及相应的驱动版本可以使用，没有GPU，程序运行时长是无法接受的。
conda install pytorch==1.10.0 torchvision==0.11.0 cudatoolkit=11.3 -c pytorch -c conda-forge
```
### opencv-python 安装
- 在对特征空间的修改中需要对马里奥游戏传回的图像进行处理，代码中使用的是 OpenCV 工具包，安装方法如下
```bash
pip install opencv-python
```
### Baseline 代码获取
- 这次课程专门创建了 DI-advanture 仓库作为算法 baseline，推荐通过以下方式获取：
```bash
git clone https://github.com/opendilab/DI-adventure
```
如果出现网络问题，也可以直接去到 DI-advanture 的仓库手动下载后解压。这样做的缺陷是需要手动初始化 git 与设置远端仓库地址：
```bash
# 如果您是通过手动解压的方式才需要执行以下内容
git init
git add * && git commit -m 'init repo'
git remote set-url origin https://github.com/opendilab/DI-adventure.git
```
推荐使用 git 作为代码管理工具，记录每一次的修改，推荐 [git 教程](https://www.liaoxuefeng.com/wiki/896043488029600)。
### 强化学习库 DI-engine 安装
- 由于这次大作业的目标不是强化学习算法，因此代码中使用了开源强化学习库 DI-engine 作为具体的强化学习算法实现，安装方法如下：
```bash
# clone主分支到本地
git clone https://github.com/opendilab/DI-engine.git
cd DI-engine
# 查看远程分支
git branch -a
# 使用 7f8c53ec6c0f3b7552cc144803422ca96d8da36e 的commit id。
git checkout 7f8c53ec6c0f3b7552cc144803422ca96d8da36e
pip install -e .
```
- (OPTIONAL)由于DI-adventure在不断更新，如果您目前使用的是老版本的DI-adventure，可能需要通过以下方式同步更新：
```bash
# 1. 切换到正确的DI-engine版本：
cd DI-engine
git checkout 7f8c53ec6c0f3b7552cc144803422ca96d8da36e
# 2. 更新DI-adventure
cd DI-adventure
# 确认'origin'指向远端仓库‘git@github.com:opendilab/DI-adventure.git’
git remote -v
# 以下这步如果出现各种例如merge conflict问题，可以借助互联网或咨询助教帮助解决。
# 或者直接重新安装DI-adventure，注意保存自己的更改。
git pull origin main
```
- 修改 gym 版本
```bash
# DI-engine这里可能会将gym版本改为0.25.2，需要手动改回来
pip install gym==0.25.1
```
## 2. Baseline 代码运行
- 项目结构
```bash
.
├── LICENSE
├── mario_dqn                               --> 本次大作业相关代码：利用DQN算法训练《超级马里奥兄弟》智能体
│   ├── assets
│   │   ├── dqn.png                         --> 流程示意图
│   │   └── mario.gif                       --> mario游戏gif示意图
│   ├── baseline_task_pipeline.py           --> 智能体训练入口，包含配置文件以及训练的逻辑
│   ├── evaluate.py                         --> 智能体评估函数
│   ├── __init__.py                         
│   ├── middleware.py                       --> 中间件，可以直接默认使用该文件
│   ├── model.py                            --> 神经网络结构定义文件
│   ├── policy.py                           --> 策略逻辑文件，包含经验收集、智能体评估、模型训练的逻辑
│   ├── README.md
│   ├── requirements.txt                    --> 项目依赖目录
│   └── wrapper.py                          --> 各式各样的装饰器实现
└── README.md
```

- 神经网络结构
![](assets/dqn.png)
- 代码运行
```bash
cd DI-adventure/mario_dqn
# 对于每组实验，推荐设置三个种子（seed）进行实验
python3 -u baseline_task_pipeline.py -s 0
```
## 3. 智能体性能评估
目前有两种方式：
1. tensorboard 查看训练过程中的曲线
- 首先安装 tensorboard 工具：
```bash
pip install tensorboard
```
- 查看训练日志：
```bash
tensorboard --logdir <exp_dir> --bind_all
```
tensorboard 中指标含义如下
- basic/eval_episode_return_mean：平均每局游戏（episode）所能获取的回报随着与环境交互的步数（step）的变化情况，一般1-1关卡2500分以上就是成功通关；
- basic/eval_episode_discount_return_mean：平均每局游戏（episode）所能获取的折扣回报随着与环境交互的步数（step）的变化情况；
- basic/exploration_epsilon：DQN在采集数据时，使用的是 $\epsilon$-greedy 算法，$\epsilon$ 随着与环境交互的步数（step）的变化情况，一般会是逐步下降；
- basic/train_cur_lr：神经网络学习率的变化情况；
- basic/train_q_value：训练过程中，Q-network 预测的Q值随着与环境交互的步数（step）的变化情况，大趋势会是逐步上升；
- basic/train_target_q_value：baseline 代码使用了 target-Q network 来稳定训练，这个是 target Q-network 预测的Q值随着与环境交互的步数（step）的变化情况，应该会和 Q-network 的变化趋势接近；
- basic/train_total_loss：神经网络训练过程中的损失（loss）随着 step 的变化情况；
- eval_replay_videos：评估时的渲染结果；
- eval_q_value_distribution：评估时Q值的分布随着环境交互步数的变化情况。

总体而言，目标是在尽可能少的环境交互步数能达到尽可能高的回报。

2. 对智能体性能进行评估，并保存录像：
```bash
python3 -u evaluate.py -s <SEED> -ckpt <CHECKPOINT_PATH> -rp <REPLAY_PATH>
```
- 建议先一个种子（seed）验证智能体能否正确运行，然后增加到3个种子验证智能体性能的稳定性。
- 此外该命令还会保存评估时的游戏录像（请确保您的 ffmpeg 软件可用），以供查看。

## 4. 特征处理
- 包括对于观测空间（observation space）、动作空间（action space）和奖励空间（reward space）的处理；
- 这一部分主要使用 wrapper 来实现，什么是 wrapper 可以参考：
    1. [如何自定义一个 ENV WRAPPER](https://di-engine-docs.readthedocs.io/zh_CN/latest/04_best_practice/env_wrapper_zh.html)
    2. [Gym Documentation Wrappers](https://www.gymlibrary.dev/api/wrappers/)

可以对以下特征空间更改进行尝试：
### 观测空间（observation space）
- 增加图像输入信息密度（将游戏版本从`v0`->`v1`）
- 堆叠四帧作为输入（`FrameStackWrapper(env, n_frames=4)`即可，注意同时更改config中的`obs_space=[4, 84, 84]`）
- 图像降采样（尝试游戏版本`v2`、`v3`的效果）
### 动作空间（action space）
- 动作简化（将 SIMPLE_ACTION 替换为 `[['right'], ['right', 'A']]`，同时更改 `action_shape=2`） 
- 增加动作的多样性（将 `SIMPLE_ACTION` 替换为 `COMPLEX_MOVEMENT`，同时更改 `action_shape=12`）
- 粘性动作 sticky action（给环境添加 `StickyActionWrapper`，方式和其它自带的 wrapper 相同，即`lambda env: StickyActionWrapper(env)`）
### 奖励空间（reward space）
- 尝试给予金币奖励（给环境添加 `CoinRewardWrapper`，方式和其它自带的 wrapper 相同）
- 稀疏 reward，只有死亡和过关才给reward（给环境添加 `SparseRewardWrapper`，方式和其它自带的 wrapper 相同）
# 更新计划
- [ ] 提供更多的wrapper以供尝试
- [ ] 分析范例（Class Activation Mapping + 光流等）
> 后续更新会在群里进行通知
# 对于大作业任务书的一些补充说明：
- “3.2【baseline 跑通】（3）训练出能够通关简单级别关卡（1-1 ~~，1-2~~ ）的智能体”。 考虑到算力等因素，大家只需要关注关卡1-1即可。