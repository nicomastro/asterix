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
    aligned_image = align_face(filepath=image_path, predictor=predictor,output_size=1024)
    print(f"Finished running alignment on image: {image_path}")
    return aligned_image
experiment_type = 'faces'

EXPERIMENT_DATA_ARGS = {
    "faces": {
        "model_path": "./pretrained_models/hyperstyle_ffhq.pt",
        "w_encoder_path": "./pretrained_models/faces_w_encoder.pt",
        "transform": transforms.Compose([
            transforms.CenterCrop((825,625)),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
            #transforms.Normalize([0.5401, 0.4144, 0.4061], [0.2081, 0.1738, 0.1755])])
    },
  
}

EXPERIMENT_ARGS = EXPERIMENT_DATA_ARGS[experiment_type]

#Load HyperStyle Model
model_path = EXPERIMENT_ARGS['model_path']
net, opts = load_model(model_path, update_opts={"w_encoder_checkpoint_path": EXPERIMENT_ARGS['w_encoder_path']})
print('Model successfully loaded!')
pprint.pprint(vars(opts))


p = Path('../datasets/OSU/OSU_crop/')
pictures = os.listdir(p)
#
print('longitud = ', len(pictures))
n = len(pictures)
#imgs = []

for k, pic in enumerate(sorted(pictures)):
 
    input_image = run_alignment(str(os.path.join(p, pic)))
    
    #input_image.resize((256, 256))
    n_iters_per_batch = 5 #@param {type:"integer"}
    opts.n_iters_per_batch = n_iters_per_batch
    opts.resize_outputs = False 


    #Run Inference
    img_transforms = EXPERIMENT_ARGS['transform']
    transformed_image = img_transforms(input_image)
    #imgs.append(transformed_image)

    with torch.no_grad():
        y_hat1, latent1, weights_deltas1, codes1= run_inversion(transformed_image.unsqueeze(0).cuda(), 
                                                              net, opts,return_intermediate_results=False)
     
    print('----------------------------------------------------------------------------')    
    print('saving npz file ', k) 
    np.savez('Directions/all_img_cropped_norm/%s.npz' % (pic.replace('.jpg', '')), w=latent1.reshape([1,18,512]).cpu())
    


#all_imgs = torch.stack(imgs)
#all_imgs = all_imgs.view(all_imgs.size(0), all_imgs.size(1), -1)
#print(all_imgs.mean(2).sum(0)/n)
#print(all_imgs.var(2).sum(0)/n)

        