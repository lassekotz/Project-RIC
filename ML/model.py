import torch
from torch import nn
import numpy as np
import pandas as pd

#df = pd.read_csv('../data/images.csv')
#print("shape is " + str(df.shape))
#print(type(df.loc[0]))
#print(df.head(2))

'''
#df = pd.read_csv('../data/images.csv', sep=',', header=None)
#print(type(df.loc[0]))

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Conv2d(1, 1, (2,2))
        #self.layer2
        #self.layer3

    def forward(self, x):
        x = self.layer1(x)

        return x


#model = Model()
#model.forward(x)
'''