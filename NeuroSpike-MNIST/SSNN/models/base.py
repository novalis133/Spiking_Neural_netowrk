import torch.nn as nn
import snntorch as snn

class BaseSNN(nn.Module):
    def __init__(self, config, spike_grad):
        super().__init__()
        self.config = config
        self.network = self._build_network(spike_grad)

    def _build_network(self, spike_grad):
        return nn.Sequential(
            nn.Conv2d(1, self.config.conv1_channels, self.config.kernel_size),
            nn.MaxPool2d(2),
            self._create_neuron(spike_grad),
            nn.Conv2d(self.config.conv1_channels, self.config.conv2_channels, self.config.kernel_size),
            nn.MaxPool2d(2),
            self._create_neuron(spike_grad),
            nn.Flatten(),
            nn.Linear(self.config.conv2_channels * 4 * 4, self.config.output_size),
            self._create_neuron(spike_grad, output=True)
        )

    def _create_neuron(self, spike_grad, output=False):
        raise NotImplementedError

    def forward(self, x):
        return self.network(x)