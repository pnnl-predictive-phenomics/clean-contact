import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, hidden_dim, out_dim, device, dtype, drop_out=0.1):
        super(Model, self).__init__()
        self.hidden_dim1 = hidden_dim
        self.out_dim = out_dim
        self.drop_out = drop_out
        self.device = device
        self.dtype = dtype

        self.fc1_1 = nn.Linear(2048, hidden_dim, dtype=dtype, device=device)
        self.fc1_2 = nn.Linear(2560, hidden_dim, dtype=dtype, device=device)
        self.ln1 = nn.LayerNorm(hidden_dim, dtype=dtype, device=device)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim,
                             dtype=dtype, device=device)
        self.ln2 = nn.LayerNorm(hidden_dim, dtype=dtype, device=device)
        self.fc3 = nn.Linear(hidden_dim, out_dim, dtype=dtype, device=device)
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, esm_x, con_x):
        # esm_x, con_x = x[:, :2560], x[:, 2560:]
        x = self.dropout(self.ln1(self.fc1_1(con_x) + self.fc1_2(esm_x)))
        # x = self.dropout(self.ln1(self.fc1(x)))
        x = torch.relu(x)
        x = self.dropout(self.ln2(self.fc2(x)))
        x = torch.relu(x)
        x = self.fc3(x)
        return x