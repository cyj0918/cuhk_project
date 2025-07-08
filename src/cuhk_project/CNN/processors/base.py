import torch

class Base:
    def __init__(self, config: dict):
        self.config = config

    def process(self, image: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    def save_result(self, image: torch.Tensor, path: str):
        """Method of Saving Results"""
        torch.save(image, path)