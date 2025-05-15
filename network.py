import torch
import torch.nn.functional as F
from torch import nn
import math

class feat2Embed(nn.Module):
    def __init__(self):
        super(feat2Embed, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(9139, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return encoded

class PLMC(nn.Module):
    def __init__(self, protein_hf_dim, hid_embed_dim, num_heads, dropout, max_length=1000):
        super(PLMC, self).__init__()

        self.dropout = dropout 
        self.embed_dim = hid_embed_dim 
        self.protein_hf_dim = protein_hf_dim 

        self.feat2_embed = feat2Embed()
        self.protein_hf_layer = nn.Linear(self.protein_hf_dim, self.embed_dim)
                                                                               
        self.protein_hf_layer_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.protein_hf_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)

        self.total_layer = nn.Linear(self.embed_dim, self.embed_dim)
        self.total_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.total_layer_1 = nn.Linear(self.embed_dim, 2) # to się głupio nazywa
        self.softmax = nn.Softmax(dim=1)

    def forward(self, protein_hf, device):

        p_feature2 = self.feat2_embed(protein_hf.to(torch.float32)) 
        p_feature2 = F.relu(self.protein_hf_bn(self.protein_hf_layer(p_feature2)))  
        p_feature2 = F.dropout(p_feature2, training=self.training, p=self.dropout)
        p_feature2 = self.protein_hf_layer_1(p_feature2) 

        p_feature = p_feature2 # (o_O)

        p_feature = F.relu(self.total_bn(self.total_layer(p_feature)))
        p_feature = F.dropout(p_feature, training=self.training, p=self.dropout)
        p_feature = self.total_layer_1(p_feature)
        
        probs = self.softmax(p_feature)
        
        return probs