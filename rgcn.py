import torch
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

class RGCN(torch.nn.Module):
    def __init__(self,num_nodes,num_relations,in_channels=64,hidden=128,out_channels=64):
        super().__init__()
        self.emb=torch.nn.Embedding(num_nodes,in_channels)
        self.conv1=RGCNConv(in_channels,hidden,num_relations)
        self.conv2=RGCNConv(hidden,out_channels,num_relations)

    def forward(self,x,edge_index,edge_type):
        x=self.emb.weight
        x=F.relu(self.conv1(x,edge_index,edge_type))
        x=F.dropout(x,p=0.3,training=self.training)
        x=self.conv2(x,edge_index,edge_type)
        return x
