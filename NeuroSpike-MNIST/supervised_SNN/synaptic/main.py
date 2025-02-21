# imports
import csv
from datetime import datetime, time
from snntorch import functional as SF
from snntorch import utils
from snntorch import surrogate

import snntorch as snn
from snntorch import backprop
from tqdm import tqdm
from models import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# from early_stopping import EarlyStopper

#################
# SNN parameter #
#################
num_inputs = 784
num_output = 10
num_hidden = 1000
batch_size = 128
num_steps = 25

dtype = torch.float
device = torch.device("cuda") #if torch.cuda.is_available() else torch.device("cpu")
####################
# data pre-process #
####################
data_path = "./data/F_mnist"
transform = transforms.Compose(
    [
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,)),
    ]
)

mnist_train = datasets.FashionMNIST(
    data_path, train=True, download=True, transform=transform
)
mnist_test = datasets.FashionMNIST(
    data_path, train=False, download=True, transform=transform
)
# reduce datasets by 10x to speed up training
subset = 10
utils.data_subset(mnist_train, subset)
utils.data_subset(mnist_test, subset)

# Create DataLoaders
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True,
                          num_workers=14, pin_memory=True,
                          )
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True,
                         num_workers=14, pin_memory=True )

##########################
# Surrogate GD functions #
# based on section 5.2.2 #
##########################
# default slope=25 & beta=1
FastSigmoid = surrogate.fast_sigmoid()
Sigmoid = surrogate.sigmoid()
SparseFastSigmoid = surrogate.SFS()
SpikeRateEscape = surrogate.spike_rate_escape()
StochasticSpikeOperator = surrogate.SSO()
StraightThroughEstimator = surrogate.straight_through_estimator()
ATan = surrogate.atan()
SGs = [FastSigmoid, Sigmoid, SparseFastSigmoid, SpikeRateEscape,
       StochasticSpikeOperator, StraightThroughEstimator, ATan]
# spike_grad = FastSigmoid
beta = 0.5  # for all neurons models


snn.slope = 50


# define early stopping function
early_stopper = EarlyStopper(patience=10, min_delta=10)


if __name__ == "__main__":
    #torch.cuda.empty_cache()
    # net = Net_Leaky()
    # test_net = Model()
    # net.half()
    # net.train()
    # test_net.eval()

    for i in SGs:
        ##############################################
        #  Initialize Networks without cuda problem  #
        #  based on the neuron list in section 5.2.1 #
        ##############################################
        Net_Synaptic = nn.Sequential(nn.Conv2d(1, 12, 5),
                                     nn.MaxPool2d(2),
                                     snn.Synaptic(alpha=alpha, beta=beta, spike_grad=i, init_hidden=True),
                                     nn.Conv2d(12, 64, 5),
                                     nn.MaxPool2d(2),
                                     snn.Synaptic(alpha=alpha, beta=beta, spike_grad=i, init_hidden=True),
                                     nn.Flatten(),
                                     nn.Linear(64 * 4 * 4, 10),
                                     snn.Synaptic(alpha=alpha, beta=beta, spike_grad=i, init_hidden=True,
                                                  output=True)
                                     ).to(device)

        optimizer = torch.optim.Adamax(Net_Synaptic.parameters(), lr=1e-2, betas=(0.9, 0.999), eps=1e-07)

        loss_fn = SF.ce_rate_loss()

        # Time
        date = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        csv_file = (
                str(retrieve_name(i))
                +'_Synaptic_Adamax_'
                + "_BS_"
                + str(batch_size)
                + '_BPTT'
                + ".csv"
        )
        with open(csv_file, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["SGD", "SNN arch", "Epoch", "Train Loss", "Train Acc", "Test Acc"])

        epochs = 100
        test_acc_hist = []
        train_acc_hist = []
        for epoch in range(epochs):
            # training loop
            # get the loss based on the time variant backpropagation algorithms
            # alterantive change
            avg_loss = backprop.BPTT(Net_Synaptic, train_loader, optimizer=optimizer, criterion=loss_fn,
                                     num_steps=num_steps, time_var=False, device=device)

            # print(f"Epoch {epoch}, Train Loss: {avg_loss.item():.2f}")
            # train set accuracy
            train_acc = batch_accuracy_Synaptic(test_loader, Net_Synaptic, num_steps)
            train_acc_hist.append(train_acc)

            # print(f"Epoch {epoch}, Train Acc: {train_acc * 100:.2f}%\n")
            # Test set accuracy
            test_acc = batch_accuracy_Synaptic(test_loader, Net_Synaptic, num_steps)
            test_acc_hist.append(test_acc)

            print(
                f"SNN: {'Synaptic'}, SGD: {str(retrieve_name(i))} ,Epoch {epoch}, Train Loss: {avg_loss.item():.2f}, Train Acc: {train_acc * 100:.2f}%, Test Acc: {test_acc * 100:.2f}%")

            with open(csv_file, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([str(retrieve_name(i)), 'Synaptic', epoch,
                                 avg_loss.item(), train_acc * 100, test_acc * 100])

            if early_stopper.early_stop(round(avg_loss.item() ,4)):
                break
        #torch.cuda.empty_cache()
        print('Training using', str(retrieve_name(i)), 'is done')
