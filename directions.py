# See notebook directions_lr
import numpy as np
from sklearn.linear_model import LinearRegression
from pathlib import Path
import glob
import pickle
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil
from sklearn.model_selection import KFold

def chunks(xs, n):
    n = max(1, n)
    return np.array([xs[i:i+n] for i in range(0, len(xs), n)])
    
def get_key(fp):
    filename = os.path.splitext(os.path.basename(fp))[0]
    int_part = filename.split()[0]
    return int(int_part)

gestos_npz = sorted(Path('Directions/all_img_cropped').glob('*.npz'), key=get_key)

# del latente de 18x512, me quedo solo con la primera fila porque son todos iguales!!
npz = np.array([(torch.tensor(np.load(x)['w'])).squeeze()[0,:].numpy() for x in gestos_npz])
emotion = chunks(npz,230) 
X_neutral = emotion[0]


# working! basic directions
for k in range(1,len(emotion)):
    score = 0
    path = f'models/m8/REG{k}/'
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path) 
    
    for j in range(512):
        X = X_neutral[:,j].reshape(-1,1)
        y = emotion[k][:,j]
        reg = LinearRegression().fit(X, y)
        score+= reg.score(X,y)
        
        filename = f'models/m8/REG{k}/LR_model{k}_{j}.sav'
        pickle.dump(reg, open(filename, 'wb'))
    print(k, 'score = ', score/512)


# mixed directions

kf = KFold(n_splits=20, random_state=None, shuffle=True)

# expresiones = [['02', '08', '09'], ['03', '10', '11', '12', '13'], ['04','14','15','16'], 
#                ['05','17','18'], ['06'], ['07','19']]

# for k, e in enumerate([[1,7,8],[2,9,10],[3,14,9],[4,16,17],[5,11,14],[6,18,15]]):
#     y = np.concatenate(emotion[e],axis=0) #690x512'
#     for i, (train_index, test_index) in enumerate(kf.split(y)):
#         score = 0
#         path = f'models/m8/m8_{i}/REG{e[0]}/'
#         if os.path.exists(path): shutil.rmtree(path)
#         os.makedirs(path) 
#         for j in range(512):
#             X_k = X_neutral[test_index%230][:,j].reshape(-1,1)
#             y_k = y[test_index][:,j]
#             reg = LinearRegression().fit(X_k, y_k)
#             score+= reg.score(X_k,y_k)
#             filename = f'models/m8/m8_{i}/REG{e[0]}/LR_model{k}_{j}.sav'
#             pickle.dump(reg, open(filename, 'wb'))
#         print(k, 'score = ', score/512)
