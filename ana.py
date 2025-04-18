import time

from models.DGM.data_load import load_data
from models.DGM.metrics import *
from models.DGM.model import *

import torch
import torch.nn.functional as F
from pprint import pprint

area = '01001'

demos = np.load(f"data/{area}/demos.npy")
print(demos.shape)
pois = np.load(f"data/{area}/pois.npy")
print(pois.shape)
feat = np.concatenate([demos, pois], axis=1)

feat_o, feat_d = feat, feat
feat_o = feat_o.reshape([feat_o.shape[0], 1, feat_o.shape[1]]).repeat(feat_o.shape[0], axis=1)
feat_d = feat_d.reshape([1, feat_d.shape[0], feat_d.shape[1]]).repeat(feat_d.shape[0], axis=0)

dis = np.load(f"data/{area}/dis.npy")
dis = dis.reshape([dis.shape[0], dis.shape[1], 1])

x = np.concatenate([feat_o, feat_d, dis], axis=2) # shape: (n, n, d)
y = np.load(f"data/{area}/od.npy") # shape: (n, n)


from models.OT_R.model import RIOT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")