import torch.nn as nn

class LSTMHead(nn.Module):
    def __init__(self, in_dim, hid_dim=128, num_layers=1, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hid_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hid_dim, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])


class TransformerHead(nn.Module):
    def __init__(self, in_dim, n_heads=2, hid_dim=128, num_layers=2, num_classes=2):
        super().__init__()
        self.projection = nn.Linear(in_dim, hid_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hid_dim, 
            nhead=n_heads,   
            dim_feedforward=hid_dim*4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(hid_dim, num_classes)

    def forward(self, x):
        x = self.projection(x) 
        out = self.transformer(x) 
        return self.fc(out[:, -1])
import torch.nn as nn

class LSTMHead(nn.Module):
    def __init__(self, in_dim, hid_dim=128, num_layers=1, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hid_dim, num_layers, batch_first=True)
        self.fc   = nn.Linear(hid_dim, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])


class TransformerHead(nn.Module):
    def __init__(self, in_dim, n_heads=2, hid_dim=128, num_layers=2, num_classes=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_dim,
            nhead=n_heads,
            dim_feedforward=hid_dim,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc          = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        out = self.transformer(x)  
        return self.fc(out[:, -1])
