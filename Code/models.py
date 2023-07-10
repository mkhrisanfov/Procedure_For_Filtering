import torch
import torch.nn as nn

class CNN1D(nn.Module):
    """CNN1D class from the original paper implemented in Python with PyTorch"""
    def __init__(self):
        super(CNN1D, self).__init__()
        self.enc0 = nn.Sequential(
            nn.Conv1d(33, 300, 6, 1),
            nn.ReLU(),
            nn.Conv1d(300, 300, 3, 1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(337, 600),
            nn.ReLU(),
            nn.Linear(600, 1),
            nn.Identity(),
        )

    def forward(self, x, col):
        x = self.enc0(x)
        # dimensions of all training tensors are 33*256 so using 
        # sum instead of average pooling for each leayer makes
        # much more sense from performance standpoint without any 
        # meaningful sacrifices in overall architecture
        x = x.sum(dim=2,keepdim=True).squeeze()
        x = torch.cat([x, col], dim=1)
        return self.fc(x)
    

class CNN2D(nn.Module):
    def __init__(self):
        super(CNN2D, self).__init__()
        # Bx16x64x64
        self.conv0=nn.Sequential(
            nn.Conv2d(16,32,3,1,1),
            nn.Conv2d(32,32,3,1,1),
            nn.ReLU(),
        )# Bx32x64x64
        
        self.conv1=nn.Sequential(
            nn.MaxPool2d(2,2),
            nn.Conv2d(48,64,3,1,1),
            nn.Conv2d(64,64,3,1,1),
            nn.ReLU(),
        )# Bx64x32x32
        
        self.conv2=nn.Sequential(
            nn.MaxPool2d(2,2),
            nn.Conv2d(96,128,3,1,1),
            nn.Conv2d(128,128,3,1,1),
            nn.ReLU(),
        )# Bx128x16x16
        
        self.fc0=nn.Sequential(
            nn.Linear(16+32+64+128+37,600),
            nn.ReLU(),
            nn.Linear(600,1),
            nn.Identity()
        )
    def forward(self,x,col):
        #add sum of initial values
        c_all=x.sum(dim=[2,3],keepdim=True).squeeze()
        #conv0 result 64x64
        y=self.conv0(x)
        #add sum of conv0 result
        c_all=torch.cat([c_all,y.sum(dim=[2,3],keepdim=True).squeeze()],dim=1)
        #concat input and conv_0 result => MaxPool=> conv1 32x32
        x=self.conv1(torch.cat([x,y],dim=1))
        #add sum of conv1 result
        c_all=torch.cat([c_all,x.sum(dim=[2,3],keepdim=True).squeeze()],dim=1)
        #concat conv1 res and conv0 res => MaxPool => conv2
        z=nn.MaxPool2d(2,2)(y)
        y=self.conv2((torch.cat([x,z],dim=1)))
        #add sum of conv2 result
        c_all=torch.cat([c_all,y.sum(dim=[2,3],keepdim=True).squeeze(),col],dim=1)
        return self.fc0(c_all)
    
class MLP(nn.Module):
    """MLP class from the original paper implemented in Python with PyTorch"""
    def __init__(self):
        super(MLP, self).__init__()
        self.fc_ds_0=nn.Sequential(
            nn.Linear(1604+167+37,300),
            nn.Tanh(),
            nn.Linear(300,300),
            nn.ReLU()
        )
        self.fc_fp_0=nn.Sequential(
            nn.Linear(1024,1200),
            nn.ReLU(),
        )
        self.fc_res_1=nn.Sequential(
            nn.Linear(1200,1200),
            nn.Dropout(0.05),
            nn.ReLU(),
            nn.Linear(1200,1200),
            nn.Dropout(0.05),
            nn.ReLU()
        )
        self.fc_res_2=nn.Sequential(
            nn.Linear(1200,1200),
            nn.Dropout(0.05),
            nn.ReLU(),
            nn.Linear(1200,1200),
            nn.Dropout(0.05),
            nn.ReLU()
        )
        self.fc_3=nn.Sequential(
            nn.Linear(1500,600),
            nn.ReLU(),
            nn.Linear(600,1),
            nn.Identity()
        )
    
    def forward(self,md,fp,maccs,col):
        x=self.fc_ds_0(torch.cat([md,col,maccs],dim=1))
        y=self.fc_fp_0(fp)
        z=self.fc_res_1(y)+y
        y=self.fc_res_2(z)+z
        x=torch.cat([x,y],dim=1)
        return self.fc_3(x)