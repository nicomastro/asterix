
import glob
import sys
from pathlib import Path
import torch
import numpy as np
from OSU import npz_to_img_stylemix


source_dir_img = Path('videos/S105/007/latente')
W = source_dir_img.glob('*.npz')
img_npz = [torch.tensor(np.load(x)['w']).reshape([18,512]).cuda() for x in sorted(W)]
print(len(img_npz))

source_dir_seed = Path('testeo_basico/mix/seeds/')
W = source_dir_seed.glob('*.npz')
seed_npz = [torch.tensor(np.load(x)['w']).reshape([18,512]).cuda() for x in sorted(W)]
print(len(seed_npz))
x = img_npz[0]
y = seed_npz[0]
#x[6:18] =y[6:18]
print(x.shape)
weight = [None]*26
res = npz_to_img_stylemix(x,weight)
#res.save('testeo_basico/mix/results/'+'%s.png' % (pictures[k].replace('.png', '')))
