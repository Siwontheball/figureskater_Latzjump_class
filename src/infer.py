import torch, numpy as np, sys
from model import LSTMClassifier

def predict(npy_path):
    seq = np.load(npy_path)[None,...]            # add batch dim
    x   = torch.from_numpy(seq).float()
    model = LSTMClassifier()
    model.load_state_dict(torch.load("best_model.pt", map_location="cpu"))
    model.eval()
    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(dim=1).item()
    return ["inside","right","flat"][pred]

if __name__=="__main__":
    path = sys.argv[1]
    print("Prediction:", predict(path))
