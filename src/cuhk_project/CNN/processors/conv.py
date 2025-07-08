# Convolution Process
from ..processors.base import Base
from torch import nn

class Conv(Base):
    def __init__(self, config):
        super().__init__(config)
        self.conv = nn.Conv2d(**config)
    
    def process(self, image):
        return super().process(image)