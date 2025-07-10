# Convolution Process
import torch
import torch.nn as nn
from typing import Dict
from .base import Base
from cuhk_project.utils.logger import configure_logging, get_logger

class Conv(Base):
    """Standard Conv processor, implementation of 2D convolution"""
    
    @staticmethod
    def required_config_keys() -> list:
        """Required configuration"""
        return [
            'in_channels', 
            'out_channels',
            'kernel_size',
            'stride',
            'padding'
        ]

    def __init__(self, config: Dict):
        """Initialize Conv layer
        
        Args:
            config: Including dictionary of keys:
                in_channels: Input channels
                out_channels: Output channels
                kernel_size: Size of kernel
                stride: Stride
                padding: Padding
                bias: Bias or not (default True)
                dilation: Dilation parameter (default 1)
                groups: Grouping parameter (default 1)
        """
        super().__init__(config)
        
        # Initialize Conv layer
        self.conv = nn.Conv2d(
            in_channels=config['in_channels'],
            out_channels=config['out_channels'],
            kernel_size=config['kernel_size'],
            stride=config.get('stride', 1),
            padding=config.get('padding', 0),
            bias=config.get('bias', True),
            dilation=config.get('dilation', 1),
            groups=config.get('groups', 1)
        )
        
        # Initialize weight
        self._init_weights()
        self.logger.info(f"Initialized Conv2d layer: {self.conv}")

    def _init_weights(self):
        """Xavier initialize conv weight"""
        nn.init.xavier_uniform_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0.1)

    def process(self, image: torch.Tensor) -> torch.Tensor:
        """Do the Conv process
        
        Args:
            image: Input tensors (1xCxHxW)
            
        Returns:
            Tensors after Conv
            
        Raises:
            ValueError: When input tensors are not as required
        """
        if image.dim() != 4:
            raise ValueError(f"Expected 4D tensor (1xCxHxW), got {image.dim()}D")
            
        if image.size(1) != self.conv.in_channels:
            raise ValueError(
                f"Input channels mismatch. Expected {self.conv.in_channels}, "
                f"got {image.size(1)}"
            )
            
        self.logger.info(
            f"Processing image with shape {image.shape} "
            f"using kernel {self.conv.kernel_size}"
        )
        
        return self.conv(image)

    def extra_repr(self) -> str:
        """Extra info"""
        return (
            f"in_channels={self.conv.in_channels}, "
            f"out_channels={self.conv.out_channels}, "
            f"kernel_size={self.conv.kernel_size}"
        )
