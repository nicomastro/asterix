import time
import os
import sys
import pprint
import numpy as np
import torch
import torchvision.transforms as transforms
import glob
import shutil
import pickle
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import re
import pandas as pd
sys.path.append(".")
sys.path.append("..")
from pathlib import Path
from utils.common import tensor2im
from utils.domain_adaptation_utils import run_domain_adaptation
from utils.model_utils import load_model, load_generator
from editing.face_editor import FaceEditor
from PIL import Image
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from twodlda import *
from LDA2D import * 
from PIL import Image, ImageOps
from torchvision.utils import save_image
from sklearn.neighbors import KNeighborsClassifier

def get_key(fp):
    filename = os.path.splitext(os.path.basename(fp))[0]
    int_part = filename.split()[0][len('LR_model'):]
    return int(int_part) 

def get_key_ck(fp):
    filename = os.path.splitext(os.path.basename(fp))[0]
    int_part = filename.split()[0]
    int_part = int_part.split('_')[-1]
    return int(int_part) 

def mkdirs(subject, e, direction):

    path = fr'./results/new_temp/{subject}/imagen{e}_{direction}'
    latent_path = f'./results/new_temp/{subject}/latente{e}'
   
    os.makedirs(path)
    if not os.path.exists(latent_path):
        os.makedirs(latent_path)

    return latent_path, path

def run_inversion(inputs, net, opts, return_intermediate_results=False):
    y_hat, latent, weights_deltas, codes = None, None, None, None

    if return_intermediate_results:
        results_batch = {idx: [] for idx in range(inputs.shape[0])}
        results_latent = {idx: [] for idx in range(inputs.shape[0])}
        results_deltas = {idx: [] for idx in range(inputs.shape[0])}
    else:
        results_batch, results_latent, results_deltas = None, None, None
        
    
    for iter in range(opts.n_iters_per_batch):
        y_hat, latent, weights_deltas, codes, _ = (
            net.forward(inputs,y_hat=y_hat, codes=codes,
                                weights_deltas=weights_deltas,
                                return_latents=True,resize=opts.resize_outputs,
                                randomize_noise=False,
                                return_weight_deltas_and_codes=True))
                                                                        
        if return_intermediate_results:
            store_intermediate_results((results_batch, results_latent,
                                        results_deltas,y_hat, latent, 
                                        weights_deltas))

        # Resize input to 256 before feeding into next iteration
        y_hat = net.face_pool(y_hat)
    
    if return_intermediate_results:
        return results_batch, results_latent, results_deltas

    return y_hat, latent, weights_deltas, codes

def store_intermediate_results(results_batch, results_latent, results_deltas,
                               y_hat, latent, weights_deltas):
    for idx in range(y_hat.shape[0]):
        results_batch[idx].append(y_hat[idx])
        results_latent[idx].append(latent[idx].cpu().numpy())
        results_deltas[idx].append([w[idx].cpu().numpy() if w is not None else None 
                                        for w in weights_deltas])
        
def run_alignment(image_path, output_size):
    import dlib
    from scripts.align_faces_parallel import align_face
    predictor = (dlib.shape_predictor(
        "./predictor/shape_predictor_68_face_landmarks.dat"))
    aligned_image = align_face(image_path, predictor,output_size)
    print(f"Finished running alignment on image: {image_path}")
    return aligned_image

  
def get_coupled_results(result_batch, transformed_image):
    result_tensors = result_batch[0] # there's one image in our batch
    final_rec = tensor2im(result_tensors[-1]).resize(resize_amount)
    input_im = tensor2im(transformed_image).resize(resize_amount)
    res = np.concatenate([np.array(input_im), np.array(final_rec)], axis=1)
    res = Image.fromarray(res)
    return res

def load_subject(s_id=59):
    subject_dir_set = []
    subject_lbldir_set = []
    dirs_img = glob.glob( "/data2/artgan/hyperstyle-old/videos/*/")
    dirs_emo = glob.glob( "/data2/nicolas/DB/CK/emotion/*/")

    for d in dirs_img:
        subject_dir_set.append(os.path.dirname(d))

    for d in dirs_emo:
        subject_lbldir_set.append(os.path.dirname(d))

    # Evaluar todas las expresiones para s_id y tomar los neutros
    subject = subject_dir_set[s_id]
    emotion_dirs = glob.glob(subject+"/*")
    X = []
    W = []
    for d in emotion_dirs: 
        X.append(sorted(glob.glob(d+'/img/*.png'),key=get_key_ck))
        w = np.array([np.load(npz)['w'] for npz in sorted(filter(
            lambda x: not re.search('png', x),(glob.glob(d+'/latente/*.npz'))),
            key=get_key_ck)])
        W.append(w[:,0,0,:])

    subject_label = subject_lbldir_set[s_id]
    emotion_dirs = glob.glob(subject_label+"/*")

    Y = []
    for d in emotion_dirs: 
        filename = glob.glob(d+'/*.txt')
        if len(filename) > 0:
            emo = [line.lstrip().rstrip() for line in open(filename[0], 'r')][0]   
            Y.append(int(float(emo)))
        else:
            Y.append(-1)
    ex_emotions = [-1,2,1,4,6]
    W = [W[i] for i,x in enumerate(Y) if x not in ex_emotions]
    X = [X[i] for i,x in enumerate(Y) if x not in ex_emotions]
    Y = [Y[i] for i,x in enumerate(Y) if x not in ex_emotions]
    #ck2OSU = {1:4,3:6,4:3,5:1,6:2,7:5}
    ck2OSU = {3:3,5:1,7:2}
    
    Y = (list(map(lambda x: ck2OSU[x], Y)))
    print(len(Y), len(W), len(X))
    return X, W, Y

def interpolate_samples(latent, latent_editor,img_path,weights_deltas,n,augment):
    delta=n-1
    imgs = []
    latents = []
    #k = 0
    l = 0
    subject = '/'.join(img_path.split('/')[-4:-2]) 
    path = fr'./results/new_temp/{subject}/'
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)  


    for k in range(augment):
        #for e in range(1,7):
        for e in [1,2,5,6]:
            sav = sorted(glob.glob(f'./models/m7/m7_{k}/REG{e}/*.sav'), key=get_key)
            #sav = sorted(glob.glob(f'./models/m2/REG{e}/*.sav'), key=get_key)
            sav_vect = [x for x in sav]
            vect = latent.reshape([18,512])
            inicial = torch.tensor(vect, device=torch.device('cuda')).reshape([18,512])
            vect = vect[0].reshape(1,-1)
            f = []
            for j in range(len(sav_vect)):
                m = pickle.load(open(sav_vect[j], 'rb'))
                f.append(m.predict(vect[:,j].reshape(1,-1).detach().cpu().numpy()))
            final = torch.tensor(np.asarray(f*18), device=torch.device('cuda')).reshape([18,512])
        
            latent_path, path = mkdirs(subject, e, k) # change
          
            # Interporlate intermediate latents and images
            for i in range(n):  
                if e == 6:
                    W = inicial + ((final-inicial)*(args.hdis*i/delta))
                elif e == 2:
                    W = inicial + ((final-inicial)*(args.hsad*i/delta))
                elif e in [1,5]:
                    W = inicial + ((final-inicial)*(args.hpos*i/delta))


                lat = [W.reshape([1,18,512]).float()]      
                np.savez(f'{latent_path}/{i}.npz', w=lat[0].cpu()) 
                latents.append(lat[0][0,0,:].detach().cpu().numpy())  
                weight = [None for i in range(len(weights_deltas))]   
    
                lat_image = (
                    latent_editor._latents_to_image(all_latents= lat, 
                                                    weights_deltas=weights_deltas)) 
                output_res = (lat_image[0])[0]
                output_res = ImageOps.equalize(output_res, mask=None)

                res = transforms.Compose([#transforms.CenterCrop((700,500)),    #esto cambia un monton!
                                          transforms.Resize((256, 256)),
                                          transforms.CenterCrop((180,140)), # esto cambia MUCHO
                                          transforms.Resize((256, 256)),
                                          ])(output_res)
                np_image = (np.asarray(res.convert('L')))/255
                imgs.append(np_image) 
         
                outputs_path = fr'{path}/{k}_{e}_{i}.jpg'
                res.convert('L').save(outputs_path)
                #res.save(outputs_path)

            print(f'## {e} ##################################################')
            
    return np.array(imgs), np.array(latents)

def fit_models(imgs, latents, n, augment, k,models):

    model2D = {'twoDLDASquare': twoDLDASquare, 'twoDLDA': twoDLDA}
    lda2d = model2D[models](k)

    # Mean zero data
    gmean =  np.mean(imgs,axis=0)
    #gmean = 0
    #std = np.std(imgs,axis=0)
    std = 1
    imgs = (imgs - gmean)/std
    
    # LDA (latents only): Keep only one neutral image, no augment data
    ix_lda = np.array([i*n + np.arange(1,n) for i in range(4)]).flatten()
    ix_lda = np.concatenate((np.array([0]),ix_lda))

    # Labels lda
    y_lda = np.arange(1,5).repeat(n-1)
    y_lda =  np.concatenate((np.array([0]),y_lda))

    # 2DLDA train (seq + augment ) and test imgs (just seq)
    
    #neutral_ix = np.array([augment*n + (augment*(n-1)*i) + i*1 for i in range(5)])
    neutral_ix = np.arange(n,imgs.shape[0],n)
    ix_2d = np.ones(imgs.shape[0]).astype('bool')
    ix_2d[neutral_ix] = False
    imgs_train  = imgs[ix_2d,:]
    print(imgs_train.shape)

    #ix_2d_test = np.array(
    #    [augment + i*augment*(n-1) + i*1 + np.arange(0,augment*(n-1),augment)
    #        for i in range(6)]).flatten()
    #ix_2d_test =  np.concatenate((np.array([0]),ix_2d_test))
    #imgs_test = imgs[ix_2d_test,:]
    #imgs_test = imgs_train
    
    # Labels 2DLDA
    #y_2d = np.arange(1,7).repeat((n-1)*augment)
    #y_2d =  np.concatenate((np.array([0]*augment),y_2d))

    y_2d = np.tile(np.arange(1,5).repeat((n-1)), augment)
    y_2d =  np.concatenate((np.array([0]),y_2d))
    print(y_2d)

    # Fit 2DLDA + PCA
    lda2d.fit(imgs_train,y_2d)
    Z_train = lda2d.transform(imgs_train,k)
    pca = PCA(n_components=2) 
    Z_train_flat = Z_train.reshape(Z_train.shape[0],-1)
    Z_imgs = pca.fit_transform(Z_train_flat)
    
    # Fit LDA (latents only)
    lda = LDA(n_components=2)
    #Z_lats = lda.fit(latents[ix_lda], y_lda).transform(latents[ix_lda])
    Z_lats = lda.fit(latents[ix_2d], y_2d).transform(latents[ix_2d])

    #Z_imgs = imgs_test.reshape(imgs_test.shape[0],-1)
    #return Z_imgs, Z_lats, y_lda, [lda2d, pca, lda], gmean, std

    return Z_imgs, Z_lats, y_2d, [lda2d, pca, lda], gmean, std



def fit_models_pca(imgs, latents, n, augment, k=50, models="PCA"):

    
    # LDA (latents only): Keep only one neutral image, no augment data
    ix_lda = np.array([i*n + np.arange(1,n) for i in range(3)]).flatten()
    ix_lda = np.concatenate((np.array([0]),ix_lda))

    # Labels lda
    y_lda = np.arange(1,4).repeat(n-1)
    y_lda =  np.concatenate((np.array([0]),y_lda))

    # Labels IMGS pca
    neutral_ix = np.arange(n,imgs.shape[0],n)
    ix_2d = np.ones(imgs.shape[0]).astype('bool')
    ix_2d[neutral_ix] = False
    imgs_train  = imgs[ix_2d,:]
    print(imgs_train.shape)

    y_2d = np.tile(np.arange(1,4).repeat((n-1)), augment)
    y_2d =  np.concatenate((np.array([0]),y_2d))
    print(y_2d)

    # Fit PCA
    pca = PCA(n_components=2) 
    Z_train_flat = imgs_train.reshape(imgs_train.shape[0],-1)
    Z_imgs = pca.fit_transform(Z_train_flat)
    
    # Fit LDA (latents only)
    lda = LDA(n_components=2)
    Z_lats = lda.fit(latents[ix_2d], y_2d).transform(latents[ix_2d])

    return Z_imgs, Z_lats, y_2d, [None, pca, lda], 0, 1


def plot_spaces(Zs, img_path,y):
    subject = '/'.join(img_path.split('/')[-4:-2]) 
    path = fr'./results/new_temp/{subject}'
    f, axs = plt.subplots(1, 1, figsize=(5, 5), sharex=True)
    Z_imgs = np.concatenate((Zs[0],Zs[2])) 
    y2 = np.concatenate((y,7*np.ones((Zs[2].shape[0]))))

    # agregar ln reg
    ix2label = {0:'Neutral', 1:'Happy',2:'Sad',3:'Fearful',4:'Angry',
                5:'Surprised',6:'Disgust',7:'New'}
    #ix2label = {0:'Neutral', 1:'Happy',2:'Surprised',3:'Disgust',7:'New'}           
    emotion = np.array(list(map(lambda x: ix2label[x], y2)))
    dataset = pd.DataFrame({'x1': Zs[0][:, 0], 'x2': Zs[0][:, 1], 
                            'Emotion': list(map(lambda x: ix2label[x], y))})

    sns.color_palette("Set2")
    sns.scatterplot(x=Z_imgs[:,0], y=Z_imgs[:,1], hue=emotion,palette="tab10")
    f.savefig(path+'/imgs2dlda.png')
    
    #sns_plot = sns.lmplot(data=dataset,x='x1',y='x2',line_kws={'alpha': 0.3},
    #                      ci=None,hue='Emotion',palette='Set2')
    #sns.scatterplot(x=Zs[2][:, 0], y=Zs[2][:, 1], color='gray',
    #                alpha=0.5,legend=False)
    #sns_plot.savefig(path+'/imgs2dlda.png')
  

    Z_lats = np.concatenate((Zs[1],Zs[3]))
    f, axs = plt.subplots(1, 1, figsize=(5, 5), sharex=True)
    sns.scatterplot(x=Z_lats[:,0], y=Z_lats[:,1], hue=emotion,palette='Set2')
    f.savefig(path+'/latlda.png', dpi=f.dpi)


print('starting')
experiment_type = 'faces'

# Parser
parser = argparse.ArgumentParser()
parser.add_argument('--image_path', type=str, default='./notebooks/images/brian.jpg')
parser.add_argument('--dataset', type=str, default=None)
parser.add_argument('--output_size', type=int, default=1024)
parser.add_argument('--k', type=int, default=50)
parser.add_argument('--augment', type=int, default=1)
parser.add_argument('--hsad', type=float, default=1)
parser.add_argument('--hdis', type=float, default=1)
parser.add_argument('--hpos', type=float, default=1)
parser.add_argument('--frames', type=int, default=10)
parser.add_argument('--models', type=str, default='2DLDA')


args = parser.parse_args()

EXPERIMENT_DATA_ARGS = {
    "faces": {
        "model_path": "./pretrained_models/hyperstyle_ffhq.pt",
        "w_encoder_path": "./pretrained_models/faces_w_encoder.pt",
        "image_path": args.image_path,
        "transform": transforms.Compose([
            #transforms.CenterCrop((200,160)),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
        "transform_test":  transforms.Compose([
           # transforms.CenterCrop((200,160)), # esto cambia MUCHO
            transforms.CenterCrop((180,140)), # esto cambia MUCHO
            transforms.Resize((256, 256)),
            transforms.Grayscale(),
            transforms.ToTensor()
            ])
    },
}

EXPERIMENT_ARGS = EXPERIMENT_DATA_ARGS[experiment_type]

#Load HyperStyle Model
model_path = EXPERIMENT_ARGS['model_path']
net, opts = load_model(model_path,update_opts={"w_encoder_checkpoint_path": 
                                               EXPERIMENT_ARGS['w_encoder_path']})
print('Model successfully loaded!')

#-------
n_iters_per_batch = 5 
opts.n_iters_per_batch = n_iters_per_batch
opts.resize_outputs = False 
#n = 10
#augment = 1
#-------


nframes = 0
acc = 0
acc2 = 0

# 124 subjects available
for s in range(40):
    # Load image
    if args.dataset == 'CK':
        image_path, W, Y = load_subject(s)
    else:
        image_path = [EXPERIMENT_DATA_ARGS[experiment_type]["image_path"]]

    for i, seq in enumerate(image_path):
        
        # Align Image 
        neutral = seq[0]
        input_image = run_alignment(neutral, args.output_size)
        #save_image(transformed_image, 'before_imgneutral__' + str(i)+ '.png') # esta imagen no esta cropeada

        # Transform
        transformed_image = EXPERIMENT_ARGS['transform'](input_image) 
        save_image(transformed_image, 'imgneutral__' + str(i)+ '.png') # esta imagen no esta cropeada

        # Gan Inversion: Get latent vector (once every sequence)
        with torch.no_grad():
            y_hat, lat, weights_deltas, codes= (
                run_inversion(transformed_image.unsqueeze(0).cuda(), net, opts,
                              return_intermediate_results=False))
           
        # Generate 2dlda/lda spaces

        lat_editor = FaceEditor(net.decoder)
        imgs, latents = interpolate_samples(lat,lat_editor,neutral,weights_deltas,args.frames,args.augment)
        if args.models in ["PCA","LDA"]:
            fit = fit_models_pca
        else:
            fit = fit_models

        Z_imgs, Z_lats, y, models, gmean, std  = fit(imgs, latents, args.frames, args.augment,args.k,args.models)
        
        # Evaluate remaining frames
        test_imgs = []
        test_latents = []
       
        if len(seq) > 1:
            m = -1 # Evaluate just the last frame from the entire sequence
            for j, frame in enumerate(seq[m:]):
                input_image = run_alignment(frame,args.output_size)

               
                input_image = ImageOps.equalize(input_image, mask=None)
                transformed_image = EXPERIMENT_ARGS['transform_test'](input_image)
                save_image(transformed_image, 'imgt__' + str(j)+ '.png')
                np_img = ((np.asarray(transformed_image)))[0]
                test_imgs.append(np_img)
            
            test_latents = W[i][m:]
            test_imgs = (((np.array(test_imgs))) - gmean)/std
            
            # Predict 2DLDA+PCA
            if models[0] is not None:
                z_temp = models[0].transform(test_imgs)
                z_temp_flat = z_temp.reshape(z_temp.shape[0],-1)
            else:
                z_temp_flat = test_imgs.reshape(test_imgs.shape[0],-1)

            Z_test_2d = models[1].transform(z_temp_flat)

            # 1-KNN 2DLDA
            print("---IMGS---")
            for z in Z_test_2d:
                D = np.sum((Z_imgs - z)**2,axis=1)
                print(y[np.argmin(D)], Y[i])
                acc += (y[np.argmin(D)] == Y[i])
                nframes +=1

            # 1-KNN LDA
            print("---LAT---")
            Z_test_1d = models[2].transform(test_latents)
            for z in Z_test_1d:
                D = np.sum((Z_lats - z)**2,axis=1)
                print(y[np.argmin(D)], Y[i])
                acc2 += (y[np.argmin(D)] == Y[i])  

            # Plot spaces
            Zs = [Z_imgs, Z_lats, Z_test_2d, Z_test_1d]
            print(Z_imgs.shape, Z_test_2d.shape)
            plot_spaces(Zs, neutral, y)

with open("results.txt", 'a') as f:
    res = (fr'Acc. Imgs = {acc/nframes} | Acc. lats = {acc2/nframes} | Total Sequences = {nframes} | k = {args.k} | n = {args.augment} | hpos = {args.hpos} | hsad = {args.hsad} | hdis = {args.hdis} | frames = {args.frames} | models = {args.models}')
    f.write(res)
    f.write('\n')
    print(res)
