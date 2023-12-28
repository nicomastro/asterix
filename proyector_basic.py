from OSU import encoder

import os
import glob
import sys

testing = ['S108','S106','S105','S115','S119']
#sys.path.insert(0, '../codigo/artgan-project/code')
#from crop import run_crop

#path_ck = "/videos"
#out_root_path = "CK/"

def main():
    i = 1
    imgs = glob.glob( f"./videos/{testing[i]}/*")
    print(testing[i])
    for img in imgs:
        #print(os.path.dirname(img))
        print(os.path.basename(img))
        #os.mkdir('./testeo_basico/'+testing[i]+'/'+os.path.basename(img)+'/hyper')
        #os.mkdir('./testeo_basico/'+testing[i]+'/'+os.path.basename(img)+'/e4e')
        #os.mkdir('./testeo_basico/'+testing[i]+'/'+os.path.basename(img)+'/normal')
        #os.mkdir('./testeo_basico/'+testing[i]+'/'+os.path.basename(img))
        encoder(f"./videos/{testing[i]}/"+os.path.basename(img)+'/img','./testeo_basico/'+testing[i]+'/'+os.path.basename(img))
        
	
if __name__ == '__main__':
	main()        
