#!/usr/bin/env python3
import argparse, os, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# detect device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

def main():
    p = argparse.ArgumentParser(description="Train violence detection")
    p.add_argument('--h5',        help='HDF5 file path (if HDF5 mode)')
    p.add_argument('--raw_dir',   help='Raw video root (if on-the-fly)')
    p.add_argument('--skip',      type=int, default=1, help='Frame skip')
    p.add_argument('--seq_len',   type=int, default=16, help='Frames per clip')
    p.add_argument('--batch',     type=int, default=32, help='Batch size')
    p.add_argument('--epochs',    type=int, default=30, help='Epochs')
    p.add_argument('--val_split', type=float, default=0.2, help='Val ratio')
    p.add_argument('--temporal',  choices=['transformer','lstm'], default='transformer')
    p.add_argument('--dynamic',   action='store_true', help='Use DynamicGCN')
    p.add_argument('--on_the_fly',action='store_true', help='On-the-fly pose extraction')
    args = p.parse_args()

    #1)Dataset
    if args.on_the_fly:
        from datasets.violence_dataset import OnTheFlyViolenceDataset as Ds
        dataset = Ds(raw_dir=args.raw_dir, seq_len=args.seq_len, skip=args.skip)
    else:
        from datasets.violence_dataset import ViolenceDataset as Ds
        dataset = Ds(h5_path=args.h5, seq_len=args.seq_len)

    N = len(dataset)
    if N == 0:
        raise RuntimeError("Empty dataset! Check your paths.")
    print(f"Dataset size: {N} samples")

    #2)Train/Val split
    val_n = int(N * args.val_split)
    train_n = N - val_n
    train_ds, val_ds = random_split(dataset, [train_n, val_n],
                                    generator=torch.Generator().manual_seed(42))
    print(f"Train/Val split: {train_n}/{val_n}")

    #3)DataLoaders
    train_loader = DataLoader(train_ds, batch_size=args.batch,
                              shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch,
                              shuffle=False, num_workers=0)

    #4)Infer feature dim
    example, _ = dataset[0]        # (T, J, 2)
    feat_dim = int(torch.prod(torch.tensor(example.shape[1:])))
    print(f"Inferred feature_dim = {feat_dim}")

    #5)Model
    if args.dynamic:
        from models.dynamic_gcn import DynamicGCN as GCN
    else:
        from models.posegcn     import PoseGCN    as GCN
    from models.temporal_heads import LSTMHead, TransformerHead

    gcn = GCN()
    head = LSTMHead(feat_dim) if args.temporal=='lstm' else TransformerHead(feat_dim)
    model = nn.Sequential(gcn, nn.Flatten(2), head).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    #6)Early stopping setup
    best_val = float('inf')
    patience = 5
    wait = 0
    os.makedirs('checkpoints', exist_ok=True)

    #7)Train loop
    for epoch in range(1, args.epochs+1):
        model.train()
        tloss = 0; tcorrect = 0
        for X,y in train_loader:
            X,y = X.to(DEVICE), y.to(DEVICE)
            out = model(X)
            loss = criterion(out, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            tloss     += loss.item()*X.size(0)
            tcorrect += (out.argmax(1)==y).sum().item()
        tloss /= train_n; tacc = tcorrect/train_n

        model.eval()
        vloss = 0; vcorrect = 0
        with torch.no_grad():
            for X,y in val_loader:
                X,y = X.to(DEVICE), y.to(DEVICE)
                out = model(X)
                l = criterion(out,y)
                vloss     += l.item()*X.size(0)
                vcorrect += (out.argmax(1)==y).sum().item()
        vloss /= val_n; vacc = vcorrect/val_n

        print(f"Epoch {epoch}/{args.epochs} "
              f"Train: loss={tloss:.4f}, acc={tacc:.4f} | "
              f"Val: loss={vloss:.4f}, acc={vacc:.4f}")

        torch.save(model.state_dict(), f"checkpoints/epoch{epoch}.pth")
        if vloss < best_val:
            best_val = vloss; wait=0
            torch.save(model.state_dict(), "checkpoints/best.pth")
        else:
            wait +=1
            if wait>=patience:
                print(f"No improvement in {patience} epochs → early stopping.")
                break

if __name__=='__main__':
    main()
