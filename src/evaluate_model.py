import argparse
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.posegcn import PoseGCN
from src.datasets.violence_dataset import ViolenceDataset
class FixedTransformerHead(torch.nn.Module):
    def __init__(self, in_dim, n_heads=2, hid_dim=128, num_layers=2, num_classes=2):
        super().__init__()
        self.projection = torch.nn.Linear(in_dim, hid_dim)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=hid_dim,
            nhead=n_heads,
            dim_feedforward=hid_dim*4,
            batch_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = torch.nn.Linear(hid_dim, num_classes)
    def forward(self, x):
        x = self.projection(x)
        out = self.transformer(x)
        return self.fc(out[:, -1])
class LSTMHead(torch.nn.Module):
    def __init__(self, in_dim, hid_dim=128, num_layers=1, num_classes=2):
        super().__init__()
        self.lstm = torch.nn.LSTM(in_dim, hid_dim, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hid_dim, num_classes)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])
def load_model(model_path, device):
    print("Loading model checkpoint...")
    checkpoint = torch.load(model_path, map_location=device)
    lstm_input_size = None
    for key, param in checkpoint.items():
        if 'lstm.weight_ih_l0' in key:
            lstm_input_size = param.shape[1]
            print(f"Detected LSTM input size: {lstm_input_size}")
            break
    
    if lstm_input_size == 34:  
        print("Creating simple pose-to-LSTM model...")
        class CustomViolenceModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = torch.nn.LSTM(34, 128, batch_first=True)
                self.fc = torch.nn.Linear(128, 2)
    
            def forward(self, x):
                batch_size, seq_len, joints, coords = x.shape
                x = x.reshape(batch_size, seq_len, joints * coords)
                out, _ = self.lstm(x)
                return self.fc(out[:, -1])
        
        model = CustomViolenceModel()
        if any('2.lstm' in k for k in checkpoint.keys()):
            print("Remapping sequential model keys...")
            new_state_dict = {}
            for k, v in checkpoint.items():
                if k.startswith('2.lstm'):
                    new_key = k.replace('2.lstm', 'lstm')
                    new_state_dict[new_key] = v
                elif k.startswith('2.fc'):
                    new_key = k.replace('2.fc', 'fc')
                    new_state_dict[new_key] = v
            model.load_state_dict(new_state_dict)
        else:
            
            model.load_state_dict(checkpoint)
        
        return model

def plot_confusion_matrix(cm, class_names, output_path='confusion_matrix.png'):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Confusion matrix saved to {output_path}")
    
def plot_metrics(report, output_path='metrics.png'):
    labels = list(report.keys())[:-3]
    metrics = ['precision', 'recall', 'f1-score']
    
    data = {}
    for metric in metrics:
        data[metric] = [report[label][metric] for label in labels]
    
    plt.figure(figsize=(12, 8))
    x = np.arange(len(labels))
    width = 0.25
    
    plt.bar(x - width, data['precision'], width, label='Precision')
    plt.bar(x, data['recall'], width, label='Recall')
    plt.bar(x + width, data['f1-score'], width, label='F1-score')
    
    plt.xlabel('Classes')
    plt.ylabel('Scores')
    plt.title('Model Performance Metrics')
    plt.xticks(x, labels)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Metrics chart saved to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--h5', required=True)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--seq_len', type=int, default=16)
    parser.add_argument('--output_dir', default='results')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = load_model(args.model, device).to(device)
    model.eval()
    
    dataset = ViolenceDataset(h5_path=args.h5, seq_len=args.seq_len)
    dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=False)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.numpy())
    
    class_names = ['NonViolence', 'Violence']
    
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)
    
    accuracy = np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    plot_confusion_matrix(cm, class_names, os.path.join(args.output_dir, 'confusion_matrix.png'))
    plot_metrics(report, os.path.join(args.output_dir, 'metrics.png'))
    
    with open(os.path.join(args.output_dir, 'results.txt'), 'w') as f:
        f.write("Classification Report:\n")
        f.write(classification_report(all_labels, all_preds, target_names=class_names))
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))
        f.write(f"\nAccuracy: {accuracy:.4f}\n")
    
    print(f"\nAll results saved to {args.output_dir}/")

if __name__ == '__main__':
    main()