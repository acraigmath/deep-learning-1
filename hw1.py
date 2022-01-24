# Adam Craig
# Deep Learning
# Fully Connected NN classifying FashionMNIST
# HW1

# Note: Much of this code is structurally built off of much of the code found
# on the PyTorch official tutorial, specifically at:
# https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
# So credit to those people kind enough to create their tutorial

import sys
from typing import List
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np


class PollutedImageDataset(Dataset):
    def __init__(self, root, train, download, transform):
        self.dataset = datasets.FashionMNIST(
            root=root, train=train, download=download, transform=transform
        )
        # create mappings for pollution
        # number of instances per class
        n = int(60000 / 10)
        # 1% of those instances to be polluted
        # which of those n will be re-labeled? (n choose n/100)
        rand_array = np.random.randn(n)
        self.one_percent = rand_array < 0.01  # create a boolean array
        # initialize dictionary to track which values have been seen
        # and what to map the next instance to
        # i.e. {actual_label: (times_seen, where_to_map_next)}
        # note it can't map to itself
        # If self.one_percent[self.label_dict[actual_label][0]] == True, relabel it to where_to_map_next,
        # then increment accordingly
        self.label_dict = {
            0: (0, 1),
            1: (0, 2),
            2: (0, 3),
            3: (0, 4),
            4: (0, 5),
            5: (0, 6),
            6: (0, 7),
            7: (0, 8),
            8: (0, 9),
            9: (0, 0),
        }

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx):
        (X, actual_label) = self.dataset.__getitem__(idx)
        ret_val = (X, actual_label)
        times_seen, next_mapping = self.label_dict[actual_label]
        if times_seen < 6000 and self.one_percent[times_seen] == True:
            ret_val = (X, next_mapping)
            if (next_mapping + 1) % 10 == actual_label:
                next_mapping = (next_mapping + 2) % 10
        self.label_dict[actual_label] = (times_seen + 1, next_mapping)
        return ret_val


class RightShiftedImageDataset(Dataset):
    def __init__(self, root, train, download, transform):
        self.dataset = datasets.FashionMNIST(
            root=root, train=train, download=download, transform=transform
        )

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx):
        (X, actual_label) = self.dataset.__getitem__(idx)
        new_X = torch.cat([X[:, -2:], X[:, :-2]], dim=1)
        return (new_X, actual_label)


class DownShiftedImageDataset(Dataset):
    def __init__(self, root, train, download, transform):
        self.dataset = datasets.FashionMNIST(
            root=root, train=train, download=download, transform=transform
        )

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx):
        (X, actual_label) = self.dataset.__getitem__(idx)
        new_X = torch.cat([X[-2:, :], X[:-2, :]], dim=0)
        return (new_X, actual_label)


def network_architecture(layers: int, activation_function):
    if layers == 2:
        return nn.Sequential(
            nn.Linear(28 * 28, 1024),
            activation_function,
            nn.Linear(1024, 1024),
            activation_function,
            nn.Linear(1024, 10),
        )
    else:
        return nn.Sequential(
            nn.Linear(28 * 28, 1024),
            activation_function,
            nn.Linear(1024, 10),
        )


def test_network(
    testing_data,
    training_data,
    lr_list: List,
    batch_list: List,
    activation_list: List,
    epochs: int,
    layers: List,
):
    for learning_rate in lr_list:
        for batch_size in batch_list:
            for activation_function in activation_list:
                for layer in layers:
                    print(
                        f"Iteration: (LR, B, FN, L) = ({learning_rate},{batch_size},{str(activation_function)}, {layer})"
                    )

                    class NeuralNetwork(nn.Module):
                        def __init__(self):
                            super(NeuralNetwork, self).__init__()
                            self.flatten = nn.Flatten()
                            self.linear_relu_stack = network_architecture(
                                layers, activation_function
                            )

                        def forward(self, x):
                            x = self.flatten(x)
                            logits = self.linear_relu_stack(x)
                            return logits

                    train_dataloader = DataLoader(training_data, batch_size=batch_size)
                    test_dataloader = DataLoader(testing_data, batch_size=batch_size)

                    model = NeuralNetwork()
                    loss_fn = nn.CrossEntropyLoss()
                    optimizer = torch.optim.SGD(
                        model.parameters(), lr=learning_rate, momentum=0
                    )

                    def train_loop(dataloader, model, loss_fn, optimizer):
                        size = len(dataloader.dataset)
                        for batch, (X, y) in enumerate(dataloader):
                            # Compute prediction and loss
                            pred = model(X)
                            loss = loss_fn(pred, y)

                            # Backpropagation
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                    def test_loop(dataloader, model, loss_fn):
                        size = len(dataloader.dataset)
                        num_batches = len(dataloader)
                        test_loss, correct = 0, 0

                        with torch.no_grad():
                            for X, y in dataloader:
                                pred = model(X)
                                test_loss += loss_fn(pred, y).item()
                                correct += (
                                    (pred.argmax(1) == y).type(torch.float).sum().item()
                                )

                        test_loss /= num_batches
                        correct /= size
                        print(
                            f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
                        )

                    for t in range(epochs):
                        print(f"Epoch {t+1}\n-------------------------------")
                        train_loop(train_dataloader, model, loss_fn, optimizer)
                        test_loop(test_dataloader, model, loss_fn)
                    print("Done!")
                    return model


def main():
    if len(sys.argv) == 1:
        print(
            "Please enter an argument for which part to complete. Ex: `python hw1.py 2`"
        )
        return
    flag = int(sys.argv[1])

    training_data = datasets.FashionMNIST(
        root="data", train=True, download=True, transform=transforms.ToTensor()
    )

    testing_data = datasets.FashionMNIST(
        root="data", train=False, download=True, transform=transforms.ToTensor()
    )

    if flag == 1:
        # Part 1
        lr_list = [0.001]
        batch_list = [30]
        activation_list = [nn.ReLU()]
        epochs = 25
        layers = [1, 2]
        test_network(
            testing_data,
            training_data,
            lr_list=lr_list,
            batch_list=batch_list,
            activation_list=activation_list,
            epochs=epochs,
            layers=layers,
        )

    if flag == 2:
        # Part 2
        lr_list = [1, 0.1, 0.01, 0.001]
        batch_list = [1, 10, 1000]
        activation_list = [nn.ReLU(), nn.Sigmoid()]
        epochs = 10
        layers = [2]
        test_network(
            testing_data,
            training_data,
            lr_list=lr_list,
            batch_list=batch_list,
            activation_list=activation_list,
            epochs=epochs,
            layers=layers,
        )

    if flag == 3:
        # Part 3
        # The best hyperparameters were determined in part 2
        lr_list = [0.001]
        batch_list = [1]
        activation_list = [nn.ReLU()]
        epochs = 25
        layers = [2]
        polluted_training_data = PollutedImageDataset(
            root="data", train=True, download=True, transform=transforms.ToTensor()
        )

        print("Polluted training set")
        training_data = polluted_training_data
        test_network(
            testing_data,
            polluted_training_data,
            lr_list=lr_list,
            batch_list=batch_list,
            activation_list=activation_list,
            epochs=epochs,
            layers=layers,
        )

        print("Regular training set")
        test_network(
            testing_data,
            training_data,
            lr_list=lr_list,
            batch_list=batch_list,
            activation_list=activation_list,
            epochs=epochs,
            layers=layers,
        )

    if flag == 4:
        # Part 4
        # The best hyperparameters were determined in part 2
        lr_list = [0.001]
        batch_list = [1]
        activation_list = [nn.ReLU()]
        epochs = 25
        layers = [2]
        right_testing_data = RightShiftedImageDataset(
            root="data", train=False, download=True, transform=transforms.ToTensor()
        )
        down_testing_data = DownShiftedImageDataset(
            root="data", train=False, download=True, transform=transforms.ToTensor()
        )

        print("Train a network...")
        model = test_network(
            testing_data,
            training_data,
            lr_list=lr_list,
            batch_list=batch_list,
            activation_list=activation_list,
            epochs=epochs,
            layers=layers,
        )

        def test_loop(dataloader, model, loss_fn):
            size = len(dataloader.dataset)
            num_batches = len(dataloader)
            test_loss, correct = 0, 0

            with torch.no_grad():
                for X, y in dataloader:
                    pred = model(X)
                    test_loss += loss_fn(pred, y).item()
                    correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            test_loss /= num_batches
            correct /= size
            print(
                f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
            )

        print("Right testing set")
        dataloader = DataLoader(right_testing_data, batch_size=1)
        test_loop(dataloader=dataloader, model=model, loss_fn=nn.CrossEntropyLoss())

        print("Down training set")
        dataloader = DataLoader(down_testing_data, batch_size=1)
        test_loop(dataloader=dataloader, model=model, loss_fn=nn.CrossEntropyLoss())


main()
