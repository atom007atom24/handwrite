import torch.nn as nn

class MLP_3Layer(nn.Module):
    def __init__(self,w1,w2,w3,w4):
        super(MLP_3Layer, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(w1, w2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(w2, w3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(w3, w4),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self, x):
        x = self.model(x)
        return x