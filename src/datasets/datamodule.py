"""Reproducible dataset setup across distributed training workers."""

import torch
from azureml.acft.image.components.olympus.core import OlympusDataModule


class BiomedParseDataModule(OlympusDataModule):
    def __init__(self, split_seed=0, **kwargs):
        super().__init__(**kwargs)
        self.split_seed = split_seed

    def setup(self, stage=None):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.split_seed)
            super().setup(stage)