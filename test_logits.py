import torch
from model import BitStateLM

ckpt = torch.load('bitstate_int8.pt', map_location='cpu', weights_only=False)
model = BitStateLM(ckpt['config'])
model.load_state_dict(ckpt['model'], strict=False)
model.eval()

with torch.no_grad():
    logits, _, _ = model(torch.tensor([[1, 2, 3]]))
    print('PYTHON_LOGITS:', logits[0, -1, :5].tolist())
    print('ARGMAX:', torch.argmax(logits[0, -1, :]).item())
