#Import Packages 
import time
import os
import sys
import pprint
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from pathlib import Path
import glob
sys.path.append(".")
sys.path.append("..")

from utils.common import tensor2im
#from utils.inference_utils import run_inversion
from utils.domain_adaptation_utils import run_domain_adaptation
from utils.model_utils import load_model, load_generator
from editing.face_editor import FaceEditor
import pickle
from OSU import encoder



def run_inversion(inputs, net, opts, return_intermediate_results=False):
    y_hat, latent, weights_deltas, codes = None, None, None, None

    if return_intermediate_results:
        results_batch = {idx: [] for idx in range(inputs.shape[0])}
        results_latent = {idx: [] for idx in range(inputs.shape[0])}
        results_deltas = {idx: [] for idx in range(inputs.shape[0])}
    else:
        results_batch, results_latent, results_deltas = None, None, None
        
    
    for iter in range(opts.n_iters_per_batch):
        y_hat, latent, weights_deltas, codes, _ = net.forward(inputs,
                                                              y_hat=y_hat,
                                                              codes=codes,
                                                              weights_deltas=weights_deltas,
                                                              return_latents=True,
                                                              resize=opts.resize_outputs,
                                                              randomize_noise=False,
                                                              return_weight_deltas_and_codes=True)
                                                              
                   
        # busca los gestos
        
        
        
        


        if "cars" in opts.dataset_type:
            if opts.resize_outputs:
                y_hat = y_hat[:, :, 32:224, :]
            else:
                y_hat = y_hat[:, :, 64:448, :]

        if return_intermediate_results:
            store_intermediate_results(results_batch, results_latent, results_deltas, y_hat, latent, weights_deltas)

        # resize input to 256 before feeding into next iteration
        if "cars" in opts.dataset_type:
            y_hat = torch.nn.AdaptiveAvgPool2d((192, 256))(y_hat)
        else:
            y_hat = net.face_pool(y_hat)
    
    if return_intermediate_results:
        return results_batch, results_latent, results_deltas

    return y_hat, latent, weights_deltas, codes


def store_intermediate_results(results_batch, results_latent, results_deltas, y_hat, latent, weights_deltas):
    print('encontradoooo',type(y_hat),y_hat.size())
    for idx in range(y_hat.shape[0]):
        results_batch[idx].append(y_hat[idx])
        results_latent[idx].append(latent[idx].cpu().numpy())
        results_deltas[idx].append([w[idx].cpu().numpy() if w is not None else None for w in weights_deltas])
        
print('starting')

def run_alignment(image_path):
    import dlib
    from scripts.align_faces_parallel import align_face
    predictor = dlib.shape_predictor("./predictor/shape_predictor_68_face_landmarks.dat")
    aligned_image = align_face(filepath=image_path, predictor=predictor)
    print(f"Finished running alignment on image: {image_path}")
    return aligned_image
experiment_type = 'faces'

EXPERIMENT_DATA_ARGS = {
    "faces": {
        "model_path": "./pretrained_models/hyperstyle_ffhq.pt",
        "w_encoder_path": "./pretrained_models/faces_w_encoder.pt",
        "image_path": "./notebooks/images/cara45.jpg",
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    },
    "cars": {
        "model_path": "./pretrained_models/hyperstyle_cars.pt",
        "w_encoder_path": "./pretrained_models/cars_w_encoder.pt",
        "image_path": "./notebooks/images/car.jpg",
        "transform": transforms.Compose([
            transforms.Resize((192, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    },
    "afhq_wild": {
        "model_path": "./pretrained_models/hyperstyle_afhq_wild.pt",
        "w_encoder_path": "./pretrained_models/afhq_wild_w_encoder.pt",
        "image_path": "./notebooks/images/afhq_wild_image.jpg",
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    }
}

EXPERIMENT_ARGS = EXPERIMENT_DATA_ARGS[experiment_type]

#Load HyperStyle Model
model_path = EXPERIMENT_ARGS['model_path']
net, opts = load_model(model_path, update_opts={"w_encoder_checkpoint_path": EXPERIMENT_ARGS['w_encoder_path']})
print('Model successfully loaded!')
pprint.pprint(vars(opts))

image_path = EXPERIMENT_DATA_ARGS[experiment_type]["image_path"]
original_image = Image.open(image_path).convert("RGB")
if experiment_type == 'cars':
    original_image = original_image.resize((192, 256))
else:
    original_image = original_image.resize((256, 256))
original_image

#Align Image 
input_is_aligned = False #@param {type:"boolean"}
if experiment_type == "faces" and not input_is_aligned:
    input_image = run_alignment(image_path)
else:
    input_image = original_image

input_image.resize((256, 256))
n_iters_per_batch = 5 #@param {type:"integer"}
opts.n_iters_per_batch = n_iters_per_batch
opts.resize_outputs = False 


#Run Inference
img_transforms = EXPERIMENT_ARGS['transform']
transformed_image = img_transforms(input_image) 

with torch.no_grad():
    tic = time.time()
    #result_batch, result_latents, _ = run_inversion(transformed_image.unsqueeze(0).cpu().numpy(),
    y_hat, latent, weights_deltas, codes= run_inversion(transformed_image.unsqueeze(0).cuda(), 
                                                          net, 
                                                          opts,
                                                          return_intermediate_results=False)
    toc = time.time()
    print('Inference took {:.4f} seconds.'.format(toc - tic))
    
# save image all_latents, weights_deltas
latent_editor = FaceEditor(net.decoder)
# create random latent vectors
print('----------------------------------------------------------------------------')


#res2 =  np.array(lat_image[0])
#res2 = Image.fromarray(lat_image[0])
#y_hat, codes = self.__get_initial_inversion(x, resize=True)

p = Path('./Directions/all/')# carpeta emocion
npz = p.glob('*.npz')
npz_vect = [torch.tensor(np.load(x)['w']).reshape([1,9216]).cuda() for x in sorted(npz)] 
vect = npz_vect[0]

inicial = torch.tensor(vect, device=torch.device('cuda')).reshape([18,512])  
for i in range(10):  
            
            W = inicial + ((final-inicial)*(i/9))#i
            lat = [W.reshape([1,18,512]).float()]  
            np.savez(f'./results/latente/{j}_{i}.npz', w=lat[0].cpu())   
            weight = [None for i in range(len(weights_deltas))]        
            lat_image = latent_editor._latents_to_image(all_latents= lat, weights_deltas = weight)           
            res = (lat_image[0])[0]    
            if(j<9): 
                outputs_path = fr'./results/imagen{j}/{i}.jpg'
            res.save(outputs_path) 
            print(f'## {j} ##################################################')



   
#p = Path('./')
#npz = p.glob('*.npz')
#npz_vect = [torch.tensor(np.load(x)['w']).reshape([18,512]).cuda() for x in sorted(npz)] 
#d = npz_vect[0]
#d = torch.tensor(d, device=torch.device('cuda'))      
#for idx, w in enumerate(latent):
#W = w+((d)*(2))
#latent = W.reshape([1,18,512])    
def get_coupled_results(result_batch, transformed_image):
    result_tensors = result_batch[0] # there's one image in our batch
    final_rec = tensor2im(result_tensors[-1]).resize(resize_amount)
    input_im = tensor2im(transformed_image).resize(resize_amount)
    res = np.concatenate([np.array(input_im), np.array(final_rec)], axis=1)
    res = Image.fromarray(res)
    return res
    
if opts.dataset_type == "cars":
    resize_amount = (256, 192) if opts.resize_outputs else (512, 384)
else:
    resize_amount = (256, 256) if opts.resize_outputs else (opts.output_size, opts.output_size)

#res = get_coupled_results(result_batch, transformed_image)
#res

  
#print('saving image')
#outputs_path = "./outputs"
#os.makedirs(outputs_path, exist_ok=True)
#res.save(os.path.join(outputs_path, os.path.basename(image_path)))  
#print('saving npz file') 

#np.savez(f'{outputs_path}/projected_w_happy.npz', w=result_latents[0])

