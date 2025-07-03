import torch, argparse
from torch import nn, optim
from dataset import get_loaders
from model import LSTMClassifier

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _ = get_loaders(
        args.processed_dir,
        batch_size=args.batch_size,
        val_frac=args.val_frac,
        test_frac=0,         # we only need train+val here
        seed=args.seed
    )
    model = LSTMClassifier().to(device)
    opt   = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    best_acc = 0.0

    for epoch in range(1, args.epochs+1):
        model.train()
        for x,y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

        # validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x,y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(dim=1)
                correct += (preds==y).sum().item()
                total   += y.size(0)
        acc = correct/total
        print(f"Epoch {epoch}: val_acc={acc:.3f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_model.pt")
            print("→ new best_model.pt saved")

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--processed-dir", default="processed")
    p.add_argument("--batch-size",    type=int,   default=16)
    p.add_argument("--epochs",        type=int,   default=30)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--val-frac",      type=float, default=0.1)
    p.add_argument("--seed",          type=int,   default=42)
    args = p.parse_args()
    train(args)
