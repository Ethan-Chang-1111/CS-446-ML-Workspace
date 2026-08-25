import torch
import torch.nn as nn
import torch.optim as optim

class LinearAE(nn.Module):
    def __init__(self, d_input: int, d_hidden: int):
        super().__init__()
        ### YOUR IMPLEMENTATION START ###
        self.layer = nn.Linear(d_input, d_hidden)
        ### YOUR IMPLEMENTATION END ###

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        ### YOUR IMPLEMENTATION START ###
        # return (x @ self.layer.weight.T)
        return (x @ self.layer.weight.T) + self.layer.bias
        ### YOUR IMPLEMENTATION END ###

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        ### YOUR IMPLEMENTATION START ###
        # return (z @ self.layer.weight)
        return (z - self.layer.bias) @ (self.layer.weight)
        # return (z - self.layer.bias) @ torch.linalg.pinv(self.layer.weight.T)
        ### YOUR IMPLEMENTATION END ###

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

def autoencode(data: torch.Tensor):
    ### YOUR IMPLEMENTATION START ###
    # Train an linear autoencoder from the provided data
    # Return the encoded components   
    centered_data = data - data.mean(dim=0, keepdim=True)

    # doubled_data = torch.cat((centered_data, centered_data), dim=0)

    model = LinearAE(d_input=data.shape[1], d_hidden=2)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)
    # Best So far, lr = 0.0001, epoch = 10000
    fin_loss = 0
    for _ in range(10000):
        y_pred = model(centered_data)
        loss = criterion(y_pred, centered_data)
        fin_loss = loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"loss: {fin_loss}")
    ae_components = model.encode(centered_data).detach()
    ### YOUR IMPLEMENTATION END ###
    return ae_components
