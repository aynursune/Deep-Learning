"""SegmentationNN"""
import torch
import torch.nn as nn

# https://pytorch.org/docs/stable/nn.html
#  upsample = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1)
#  output = upsample(h, output_size=input.size())
class SegmentationNN(nn.Module):

    def __init__(self, num_classes=23):
        super(SegmentationNN, self).__init__()

        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################

       ## self.upsample = nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1)
        
        from torchvision import models
        
        #self.conv1 = nn.Conv2d(3,32,5, padding = 2)
        #self.conv2 = nn.Conv2d(32, 64, 5, padding = 2)
        #self.conv3 = nn.Conv2d(64,128,5, padding = 2)
        #self.conv4 = nn.Conv2d(128,256,5, padding = 2)
        #self.conv5 = nn.Conv2d(256,512,5, padding = 2)
        #self.conv6 = nn.Conv2d(512,3,5, padding = 2)
        #
        #self.maxpool = nn.MaxPool2d(2,2)
        #self.avgpool = nn.AvgPool2d(2,2)
        #self.relu = nn.ReLU()
        #self.upsample = nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1)
        #self.ups = nn.Upsample(scale_factor=2, mode='nearest')
        #
        #self.fc1 = nn.Linear(250880,2048)
        #self.fc2 = nn.Linear(2048, 1024)
        #self.fc3 = nn.Linear(1024, 512)
        #self.fc4 = nn.Linear(512, 6)
        #self.fc = nn.Linear(60, 240)
        
        self.pre_alex = models.alexnet(pretrained=True).features
        self.dropout = nn.Dropout()
        self.avgpool = nn.AvgPool2d(2,2)
        self.upsample = nn.Upsample(scale_factor=80, mode='nearest')
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(256,2048, kernel_size = 1, padding = 0) 
        self.conv2 = nn.Conv2d(2048,4096, kernel_size = 1, padding = 0)
        self.conv3 = nn.Conv2d(4096,num_classes, kernel_size = 3, padding = 1)
        
        
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################

        x = self.pre_alex(x)
        x = self.dropout(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.upsample(x)
        
        
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
