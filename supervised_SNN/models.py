
import snntorch as snn
import numpy as np
import inspect
from snntorch import functional as SF
from snntorch import utils
from snntorch import surrogate
import torch
import torch.nn as nn
import torch.nn.functional as F
#################
# SNN parameter #
#################
num_inputs = 784
num_output = 10
num_hidden = 1000
batch_size = 128
#################
# SGD functions #
#################
# default slope=25 & beta=1
spike_grad_fs = surrogate.FastSigmoid()
spike_grad_T = surrogate.Triangular()
spike_grad_S = surrogate.Sigmoid()
spike_grad_SFS = surrogate.SparseFastSigmoid()
spike_grad_SRE = surrogate.SpikeRateEscape()
spike_grad_SSO = surrogate.StochasticSpikeOperator()
spike_grad_STE = surrogate.StraightThroughEstimator()
spike_grad_AT = surrogate.ATan()

beta = 0.5 # for leaky,
R = 1 # for Lapicque
C = 1.44 # for Lapicque
alpha = 0.9 # for Alpha
# shared recurrent connection for a given layer
V1 = 0.5 # for RLeaky and RSynaptic
# independent connection p/neuron
V2 = torch.rand(num_output) # for RLeaky and RSynaptic
snn.slope = 50

################################
# helper functions and classes #
################################
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def forward_pass(net, num_steps, data):
    mem_rec = []
    spk_rec = []
    utils.reset(net)  # resets hidden states for all LIF neurons in net

    for step in range(num_steps):
        spk_out, mem_out = net(data)
        spk_rec.append(spk_out)
        mem_rec.append(mem_out)

    return torch.stack(spk_rec), torch.stack(mem_rec)

def forward_pass_alpha(net, num_steps, data):
    """ for Alpha neurons, in each iteration the neuron produce 4 outputs; memberane,
    spikes, Excitatory and Inhibitory synaptic current. all are recorded, but for this research
    purpose, only spikes and membrane are used for batch accuracy calculating"""
    mem_rec = []
    spk_rec = []
    syn_exc_rec = [] # Excitatory synaptic recording
    syn_inh_rec = [] # Inhibitory synaptic recording
    utils.reset(net)  # resets hidden states for all LIF neurons in net

    for step in range(num_steps):
        spk_out, syn_exc, syn_inh, mem_out = net(data)
        spk_rec.append(spk_out)
        mem_rec.append(mem_out)
        syn_exc_rec.append(syn_exc)
        syn_inh_rec.append(syn_inh)

    return torch.stack(spk_rec), torch.stack(mem_rec)

def batch_accuracy(train_loader, net, num_steps):
    with torch.no_grad():
        total = 0
        acc = 0
        net.eval()

        train_loader = iter(train_loader)
        for data, targets in train_loader:
            data = data.to(device)
            targets = targets.to(device)
            spk_rec, _ = forward_pass(net, num_steps, data)

            acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
            total += spk_rec.size(1)

    return acc / total

def batch_accuracy_alpha(train_loader, net, num_steps):
    """ for Alpha neuron"""
    with torch.no_grad():
        total = 0
        acc = 0
        net.eval()

        train_loader = iter(train_loader)
        for data, targets in train_loader:
            data = data.to(device)
            targets = targets.to(device)
            spk_rec, _ = forward_pass_alpha(net, num_steps, data)

            acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
            total += spk_rec.size(1)

    return acc / total

def forward_pass_Synaptic(net, num_steps, data):
    """ for Synaptic, in each iteration the neuron produce 4 outputs; memberane,
    spikes and  single synaptic current. all are recorded, but for this research
    purpose, only spikes and membrane are used for batch accuracy calculating"""

    mem_rec = []
    spk_rec = []
    syn_rec = [] # it is recorded for further researchs
    utils.reset(net)  # resets hidden states for all LIF neurons in net
    # print(net(data))
    for step in range(num_steps):
        spk_out, syn_out, mem_out = net(data)
        spk_rec.append(spk_out)
        mem_rec.append(mem_out)
        syn_rec.append(syn_out)
    return torch.stack(spk_rec), torch.stack(mem_rec)

    return torch.stack(spk_rec), torch.stack(mem_rec)
def batch_accuracy_Synaptic(train_loader, net, num_steps):
    """ for Synaptic neuron"""
    with torch.no_grad():
        total = 0
        acc = 0
        net.eval()

        train_loader = iter(train_loader)
        for data, targets in train_loader:
            data = data.to(device)
            targets = targets.to(device)
            spk_rec, _ = forward_pass_Synaptic(net, num_steps, data)

            acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
            total += spk_rec.size(1)

    return acc / total
def retrieve_name(var):
        """
        Gets the name of var. Does it from the out most frame inner-wards.
        :param var: variable to get name from.
        :return: string
        """
        for fi in reversed(inspect.stack()):
            names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
            if len(names) > 0:
                return names[0]


class EarlyStopper:
    """
    adapted from https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
    """
    # function to stop training when the validation loss is not decreasing anymore
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
        if self.counter >= self.patience:
            return True
        else:
            return False