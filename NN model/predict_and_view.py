# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from model import CornerBiLSTM
# from features import extract_features

# model = CornerBiLSTM()
# model.load_state_dict(torch.load("model.pth", map_location="cpu"))
# model.eval()

# stroke = np.load("dataset_clean_intersections/strokes/stroke_0000.npy")
# features = extract_features(stroke)

# with torch.no_grad():
#     logits = model(torch.tensor(features).unsqueeze(0))
#     preds = torch.argmax(logits, dim=-1).squeeze().numpy()

# plt.plot(stroke[:,0], stroke[:,1], '-o')
# corner_idx = np.where(preds == 1)[0]
# plt.scatter(stroke[corner_idx,0], stroke[corner_idx,1], c='red', s=80)
# plt.axis('equal')
# plt.show()
