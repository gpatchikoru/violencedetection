import torch
import torch.nn as nn

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x, A):
       
        h = torch.matmul(A, x)       
        return self.fc(h)            

def normalized_adjacency(n_joints, edges):
    A = torch.eye(n_joints)
    for i, j in edges:
        A[i, j] = A[j, i] = 1.0
    D = A.sum(dim=1)
    D_inv_sqrt = torch.diag(D.pow(-0.5))
    return D_inv_sqrt @ A @ D_inv_sqrt
class PoseGCN(nn.Module):
    def __init__(self, in_ch=2, hid_ch=64, n_joints=17):
        super().__init__()
        edges = [
            (0,1),(0,2),(1,3),(2,4),
            (0,5),(0,6),(5,7),(7,9),
            (6,8),(8,10),(5,11),(6,12),
            (11,12),(11,13),(13,15),(12,14),(14,16)
        ]
        A = normalized_adjacency(n_joints, edges)
        self.register_buffer('A', A)    

        self.gcn1 = GCNLayer(in_ch,  hid_ch)
        self.gcn2 = GCNLayer(hid_ch, hid_ch)

    def forward(self, x):
        B, T, J, C = x.shape
        outs = []
        for t in range(T):
            h = x[:, t]                       
            h = torch.relu(self.gcn1(h, self.A))
            h = torch.relu(self.gcn2(h, self.A))
            outs.append(h)
        out = torch.stack(outs, dim=1)       
        return out.permute(0, 1, 3, 2)         
