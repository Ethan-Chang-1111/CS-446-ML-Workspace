import torch
import torch.nn as nn
import torch.optim as optim


class LinearAE(nn.Module):
    def __init__(self, d_input: int, d_hidden: int):
        super().__init__()
        ### YOUR IMPLEMENTATION START ###
        ...
        ### YOUR IMPLEMENTATION END ###

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        ### YOUR IMPLEMENTATION START ###
        ...
        ### YOUR IMPLEMENTATION END ###

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        ### YOUR IMPLEMENTATION START ###
        ...
        ### YOUR IMPLEMENTATION END ###

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def autoencode(data: torch.Tensor):
    ### YOUR IMPLEMENTATION START ###
    # Train an linear autoencoder from the provided data
    # Return the encoded components
    ### YOUR IMPLEMENTATION END ###
    return ae_components
