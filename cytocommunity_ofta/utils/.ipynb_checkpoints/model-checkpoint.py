import torch
from torch_geometric.nn import DenseGraphConv, dense_mincut_pool
import torch.nn.functional as F
from torch.nn import Linear

class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, Num_TCN, dropout_rate):
        super(Net, self).__init__()

        self.conv1 = DenseGraphConv(in_channels, hidden_channels)
        num_cluster1 = Num_TCN   # This is a hyperparameter.
        self.pool1 = Linear(hidden_channels, num_cluster1)
        
        self.conv3 = DenseGraphConv(hidden_channels, hidden_channels)
        
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)
        
        # Dropout layer
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, x, adj, mask=None):
        x = F.relu(self.conv1(x, adj, mask))
        #x = self.dropout(x)  # Apply dropout after the first convolution

        s = self.pool1(x)  # Here s is a non-softmax tensor.
        x, adj, mc1, o1 = dense_mincut_pool(x, adj, s, mask)

        # Save important clustering results_1.
        ClusterAssignTensor_1 = s
        ClusterAdjTensor_1 = adj

        x = self.conv3(x, adj)

        x = x.mean(dim=1)
        x = F.relu(self.lin1(x))
        x = self.dropout(x)  # Apply dropout after the first linear layer
        x = self.lin2(x)
        
        return F.log_softmax(x, dim=-1), mc1, o1, ClusterAssignTensor_1, ClusterAdjTensor_1

