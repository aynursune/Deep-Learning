import torch
import torch.nn as nn
import torch.nn.functional as F


class KeypointModel(nn.Module):

    def __init__(self):
        super(KeypointModel, self).__init__()

        #######################################################################
        # TODO: Define all the layers of this CNN, the only requirements are: #
        # 1. This network takes in a square (same width and height),          #
        #    grayscale image as input.                                        #
        # 2. It ends with a linear layer that represents the keypoints.       #
        # It's suggested that you make this last layer output 30 values, 2    #
        # for each of the 15 keypoint (x, y) pairs                            #
        #                                                                     #
        # Note that among the layers to add, consider including:              #
        # maxpooling layers, multiple conv layers, fully-connected layers,    #
        # and other layers (such as dropout or  batch normalization) to avoid #
        # overfitting.                                                        #
        #######################################################################
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size = 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1)
        #self.conv5 = nn.Conv2d(256, 512, kernel_size = 3, stride = 1, padding = 1)
        self.maxpool = nn.MaxPool2d(2,2)
        self.dropout1 = nn.Dropout2d(0.1)
        self.dropout2 = nn.Dropout2d(0.2)
        self.dropout3 = nn.Dropout2d(0.3)
        self.dropout4 = nn.Dropout2d(0.4)
        self.dropout41d = nn.Dropout(0.4)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.linear1 = nn.Linear(256*6*6, 256)
        self.linear2 = nn.Linear(256,30)
        
        #######################################################################
        #                             END OF YOUR CODE                        #
        #######################################################################

    def forward(self, x):
        #######################################################################
        # TODO: Define the feedforward behavior of this model                 #
        # x is the input image and, as an example, here you may choose to     #
        # include a pool/conv step:                                           #
        # x = self.pool(F.relu(self.conv1(x)))                                #
        # a modified x, having gone through all the layers of your model,     #
        # should be returned                                                  #
        #######################################################################
        org = x
        x = self.conv1(x)
        #print(x.size())
        x = self.maxpool(x)
        #print(x.size())
        x = self.relu(x)
        x = self.dropout1(x)  
        
        x = self.conv2(x)
        #print(x.size())
        x = self.maxpool(x)
        #print(x.size())
        x = self.relu(x)
        x = self.dropout2(x) 
        
        x = self.conv3(x)
        #print(x.size())
        x = self.maxpool(x)
        #print(x.size())
        x = self.relu(x)
        x = self.dropout3(x)  
        #print(x.size())
        
        x = self.conv4(x)
        #print(x.size())
        x = self.maxpool(x)
        #print(x.size())
        x = self.relu(x)
        x = self.dropout4(x) 
        
        x = x.view(org.size(0),-1)
        
        #print(x.size())
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout41d(x)        
        x = self.linear2(x)
        x = self.tanh(x)
        
         
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
        return x

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
