import torch
import torch.nn as nn
from models.posegcn import PoseGCN
from models.temporal_heads import TransformerHead, LSTMHead

def debug_model():
    batch_size = 4
    seq_len = 16
    n_joints = 17
    feature_dim = 2
    
    X = torch.randn(batch_size, seq_len, n_joints, feature_dim)
    
    gcn = PoseGCN(in_ch=feature_dim, hid_ch=64, n_joints=n_joints)
    
    gcn_out = gcn(X)
    print(f"GCN output shape: {gcn_out.shape}")
    flattened = torch.flatten(gcn_out, start_dim=2)
    print(f"Flattened shape: {flattened.shape}")
    feat_dim = flattened.shape[2]
    print(f"Feature dimension after flattening: {feat_dim}")
    transformer = TransformerHead(feat_dim, n_heads=2, hid_dim=128)
    transformer_out = transformer(flattened)
    print(f"Transformer output shape: {transformer_out.shape}")
    
    lstm = LSTMHead(feat_dim, hid_dim=128)
    lstm_out = lstm(flattened)
    print(f"LSTM output shape: {lstm_out.shape}")
    
    model = nn.Sequential(gcn, nn.Flatten(2), transformer)
    model_out = model(X)
    print(f"Full model output shape: {model_out.shape}")
    
    print("All shapes look good! Model should work correctly.")

if __name__ == "__main__":
    debug_model()