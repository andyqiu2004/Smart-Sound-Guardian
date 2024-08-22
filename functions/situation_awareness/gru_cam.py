import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns


class GRU_CAM_Model(nn.Module):
    def __init__(
        self, input_size, hidden_size, output_size, num_layers, bidirectional=True
    ):
        super(GRU_CAM_Model, self).__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.fc = nn.Linear(
            hidden_size * 2 if bidirectional else hidden_size, output_size
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        gru_out, _ = self.gru(x)
        if self.bidirectional:
            gru_out = (
                gru_out[:, :, : self.hidden_size] + gru_out[:, :, self.hidden_size :]
            )
        output = self.fc(gru_out[:, -1, :])
        return output, gru_out


class GRU_CAM:
    def __init__(self, model, input_size, hidden_size, num_layers, bidirectional=True):
        self.model = model
        self.gradients = None
        self.hook_handles = []
        self.register_hooks()

    def register_hooks(self):
        def forward_hook(module, input, output):
            self.feature_maps = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.hook_handles.append(self.model.gru.register_forward_hook(forward_hook))
        self.hook_handles.append(self.model.gru.register_backward_hook(backward_hook))

    def generate_cam(self, input_tensor, class_idx):
        self.model.zero_grad()
        output, _ = self.model(input_tensor)
        target = output[:, class_idx]
        target.backward()

        weights = torch.mean(self.gradients, dim=1, keepdim=True)
        cam = torch.sum(weights * self.feature_maps, dim=2)
        cam = cam.cpu().detach().numpy()
        cam = np.maximum(cam, 0)
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam

    def cleanup(self):
        for handle in self.hook_handles:
            handle.remove()


class SituationAwarenessDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "data": torch.tensor(self.data[idx], dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def train_gru_cam_model(model, dataloader, epochs=20, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        for batch in dataloader:
            data = batch["data"]
            labels = batch["label"]

            optimizer.zero_grad()
            output, _ = model(data)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()


def visualize_cam(cam, title="Class Activation Map"):
    sns.heatmap(cam, cmap="viridis")
    plt.title(title)
    plt.show()
