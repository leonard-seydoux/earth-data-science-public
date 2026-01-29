"""Model definition for the model used in the notebooks.

Please use this module to define the model you want to use in the notebooks. The
model should be a subclass of torch.nn.Module and should have a forward method
that takes a batch of images as input and returns a batch of logits as output.
"""

import torch.nn as nn


class MultiLayerPerceptron(nn.Module):
    """Fully connected neural network with one hidden layer.

    Arguments
    ---------
    input_size: int
        Size of the input layer.
    hidden_size: int
        Size of the hidden layer.
    num_classes: int
        Size of the output layer.
    """

    def __init__(self, input_size, hidden_size, n_classes):
        """This is the model instance.

        When called within another script, the MultiLayerPerceptron
        class with create a new instance of itself and execute the __init__()
        function. This is where we can create the instances that we need to
        connect later.

        Note that the graph (tree of all operations in the network) is not
        defined in the __init__() method, but in the forward() method.
        Here, we only create them and collect them in the "inner" persistent
        memory of the model, namely as attributes of the newly created object.
        """
        # Because MultiLayerPerceptron inherits from a torch.nn.Module object,
        # we need to instanciate the parent class first.
        super(MultiLayerPerceptron, self).__init__()

        # Here we define the operations that we will train later.
        self.linear_1 = nn.Linear(input_size, hidden_size)
        self.activation_1 = nn.ReLU()
        self.linear_2 = nn.Linear(hidden_size, n_classes)
        self.activation_2 = nn.Softmax(dim=1)

    def forward(self, x):
        """Forward pass.

        This function defines the forward model, that is, the graph of
        operations to perform on the input data to get the output. It is
        important that all the learnable operations are saved in the object
        self, which is the instant itself, to keep track of the updated
        parameters during the training and testing.

        Arguments
        ---------
        x: torch.Tensor
            Input data of shape (batch_size, input_size).
        """
        out = self.linear_1(x)
        out = self.activation_1(out)
        out = self.linear_2(out)
        out = self.activation_2(out)
        return out


class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(
        self,
        input_size=28 * 28,
        n_classes=10,
        hidden_size=16,
        kernel_size=5,
        stride=1,
        padding=2,
    ):
        # In the constructor we instantiate two nn.Linear modules and assign them as
        # member variables.
        super(ConvolutionalNeuralNetwork, self).__init__()

        # First convolutional layer, taking in 1 input channel (image),
        # outputting N convolutional features, with a square kernel size
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=hidden_size,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # Second convolutional layer, taking in N input channels (N features from
        # conv1), outputting M convolutional features, with a square kernel size
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_size,
                out_channels=hidden_size * 2,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # Ouput the 10 classes from the MNIST dataset
        self.fc = nn.Linear(hidden_size * 2 * 8 * 8, n_classes)
        self.out = nn.Softmax(dim=1)

    def forward(self, x):
        # Convolutional
        x = self.conv1(x)
        x = self.conv2(x)

        # Fully connected (flatten)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        out = self.out(x)

        # Return also x for visualization
        return out, x


class Conv1DSeismicClassifier(nn.Module):
    """1D Convolutional Neural Network for seismic waveform classification.

    Arguments
    ---------
    input_size: int
        Size of the input waveform (number of samples).
    n_classes: int
        Number of output classes.
    hidden_channels: int
        Number of channels in the first convolutional layer.
    kernel_size: int
        Size of the convolutional kernel.
    """

    def __init__(
        self,
        input_size=8192,
        n_classes=5,
        hidden_channels=32,
        kernel_size=11,
    ):
        super(Conv1DSeismicClassifier, self).__init__()

        # First convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
            ),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
        )

        # Second convolutional block
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_channels,
                out_channels=hidden_channels * 2,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
            ),
            nn.BatchNorm1d(hidden_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
        )

        # Third convolutional block
        self.conv3 = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_channels * 2,
                out_channels=hidden_channels * 4,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
            ),
            nn.BatchNorm1d(hidden_channels * 4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
        )

        # Calculate the size after convolutions and pooling
        # After 3 maxpool layers with kernel_size=4: input_size / (4 * 4 * 4) = input_size / 64
        self.fc_input_size = (input_size // 64) * hidden_channels * 4

        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, n_classes)
        self.activation = nn.ReLU()

    def forward(self, x):
        """Forward pass.

        Arguments
        ---------
        x: torch.Tensor
            Input data of shape (batch_size, input_size).
        """
        # Reshape input from (batch_size, input_size) to (batch_size, 1, input_size)
        # The 1 is the number of input channels
        x = x.unsqueeze(1)

        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Flatten for fully connected layer
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x
