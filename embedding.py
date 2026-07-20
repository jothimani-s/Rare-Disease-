import torch
import torch.nn as nn

class TransE(nn.Module):
    def __init__(self,n_entities,n_relations,embedding_dim=64,margin=1.0):
        super().__init__();self.entity=nn.Embedding(n_entities,embedding_dim);self.relation=nn.Embedding(n_relations,embedding_dim);self.margin=margin
    def score(self,h,r,t): return torch.norm(self.entity(h)+self.relation(r)-self.entity(t),p=2,dim=1)
    def forward(self,pos,neg): return torch.relu(self.margin+self.score(pos[:,0],pos[:,1],pos[:,2])-self.score(neg[:,0],neg[:,1],neg[:,2])).mean()