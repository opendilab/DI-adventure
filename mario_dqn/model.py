"""
神经网路模型定义
"""
from typing import Optional, Dict, List
import torch
import torch.nn as nn

from ding.model import ConvEncoder, DiscreteHead


class DQN(nn.Module):

    def __init__(
            self,
            obs_shape: List,
            action_shape: int,
            encoder_hidden_size_list: List = [32, 64, 64],
            head_hidden_size: Optional[int] = None,
            head_layer_num: Optional[int] = 1,
            activation: Optional[nn.Module] = nn.ReLU(),
            norm_type: Optional[str] = None
    ) -> None:
        super(DQN, self).__init__()
        if head_hidden_size is None:
            head_hidden_size = encoder_hidden_size_list[-1]
        self.encoder = ConvEncoder(obs_shape, encoder_hidden_size_list, activation=activation, norm_type=norm_type)
        self.head = DiscreteHead(
            head_hidden_size, action_shape, head_layer_num, activation=activation, norm_type=norm_type
        )

    def forward(self, x: torch.Tensor) -> Dict:
        """
        Examples:
            >>> model = DQN(32, 6)  # arguments: 'obs_shape' and 'action_shape'
            >>> inputs = torch.randn(4, 32)
            >>> outputs = model(inputs)
            >>> assert isinstance(outputs, dict) and outputs['logit'].shape == torch.Size([4, 6])
        """
        x = self.encoder(x)
        x = self.head(x)
        return x
