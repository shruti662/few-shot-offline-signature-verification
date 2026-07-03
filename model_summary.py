from torchsummary import summary
from siamese_network import SiameseNetwork
import torch.nn as nn

model = SiameseNetwork()

# Combine conv + fc
full_model = nn.Sequential(
    model.conv,
    nn.Flatten(),
    model.fc
)

summary(full_model, (1, 120, 160))