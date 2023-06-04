import torch
from torch import nn
from torchvision import models
from torchsummary import summary


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv_1 = torch.nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=5, padding='same')
        self.relu_1 = torch.nn.ReLU()
        self.conv_2 = torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding='same')
        self.relu_2 = torch.nn.ReLU()
        self.pool_1 = torch.nn.MaxPool1d(kernel_size=5, stride=2)

    def forward(self, x):
        identity = x
        
        out = self.conv_1(x)
        out = self.relu_1(out)
        out = self.conv_2(out)
        
        # https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L100
        out += identity
        
        out = self.relu_2(out)
        out = self.pool_1(out)
        return out


class NeuralNetwork(nn.Module):
    """
    Model architecture for training
    """

    def __init__(self, model_type, input_dimension, dataset_name):
        """
        Construct NeuralNetwork object and initialize member variables
        """        
        super(NeuralNetwork, self).__init__()

        self.model_type = model_type.lower()

        if self.model_type == 'author_proposed_network':
            self.conv_1 = torch.nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding='same')
            
            self.residual_block_1 = ResidualBlock(in_channels=32)
            self.residual_block_2 = ResidualBlock(in_channels=32)
            self.residual_block_3 = ResidualBlock(in_channels=32)
            self.residual_block_4 = ResidualBlock(in_channels=32)
            self.residual_block_5 = ResidualBlock(in_channels=32)
            
            self.flatten = torch.nn.Flatten()
            
            self.linear_1 = torch.nn.Linear(in_features=64, out_features=32)
            self.relu_1  = torch.nn.ReLU()
            self.linear_2 = torch.nn.Linear(in_features=32, out_features=32)
            
            if dataset_name == "mitbih":
                self.linear_3 = torch.nn.Linear(in_features=32, out_features=5)
            elif dataset_name == "ptbdb":
                self.linear_3 = torch.nn.Linear(in_features=32, out_features=2)
            else:
                raise Exception("The dataset name should be 'mitbih' or 'ptbdb")
            
            # Softmax is already included in BCELoss in Pytorch
            # self.softmax = torch.nn.Softmax()
        else: 
            raise Exception("The architecture name should be 'author_proposed_network'")
        
    def forward(self, x):
        x = self.conv_1(x)
        x = self.residual_block_1(x)
        x = self.residual_block_2(x)
        x = self.residual_block_3(x)
        x = self.residual_block_4(x)
        x = self.residual_block_5(x)
        
        x = self.flatten(x)

        x = self.linear_1(x)
        x = self.relu_1(x)
        x = self.linear_2(x)        
        x = self.linear_3(x)
        return x # x is in logit form


if __name__ == '__main__':
    trial_model = NeuralNetwork('author_proposed_network', 187, "mitbih")  # resnet18 #vgg16
    
    # print(trial_model.)
    
    # https://stackoverflow.com/questions/59013109/runtimeerror-input-type-torch-floattensor-and-weight-type-torch-cuda-floatte
    if torch.cuda.is_available():
        trial_model.cuda()

    print(summary(trial_model, (1,187)))
    # print(trial_model)
    # print(trial_model.residual_block_5.pool_1)
    
    # for layer in trial_model.children():
    #     if hasattr(layer, 'out_features'):
    #         print(layer.out_features)
