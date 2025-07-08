# Pipeline Control
from typing import List
from .processors.base import Base
from .utils.image_io import load_image

class ProcessingPipeline:
    def __init__(self, processors: List[Base]):
        self.processors = processors

    def run(self, image_path, output_dir):
        image = load_image(image_path)
        for i, processor in enumerate(self.processors):
            image = processor.process(image)
            processor.save_result(
                image,
                f"{output_dir}/step_{i}_{type(processor).__name__}.pt"
            )