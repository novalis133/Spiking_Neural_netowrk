from .base import BaseSNN
import snntorch as snn

class LeakySNN(BaseSNN):
    def _create_neuron(self, spike_grad, output=False):
        return snn.Leaky(
            beta=self.config.beta,
            spike_grad=spike_grad,
            init_hidden=True,
            output=output
        )

class AlphaSNN(BaseSNN):
    def _create_neuron(self, spike_grad, output=False):
        return snn.Alpha(
            alpha=self.config.alpha,
            beta=self.config.beta,
            spike_grad=spike_grad,
            init_hidden=True,
            output=output
        )

class SynapticSNN(BaseSNN):
    def _create_neuron(self, spike_grad, output=False):
        return snn.Synaptic(
            alpha=self.config.alpha,
            beta=self.config.beta,
            spike_grad=spike_grad,
            init_hidden=True,
            output=output
        )

class LapicqueSNN(BaseSNN):
    def _create_neuron(self, spike_grad, output=False):
        return snn.Lapicque(
            beta=self.config.beta,
            spike_grad=spike_grad,
            init_hidden=True,
            output=output
        )