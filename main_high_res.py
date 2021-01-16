import os
import numpy as np
from PIL import Image
from ISR.models import RDN, RRDN


def generate_single_img(input_img, output_img, model, times=3):
    img = Image.open(input_img)
    img = img.convert("RGB")
    for i in range(times):
        lr_img = np.array(img)
        sr_img = model.predict(lr_img)
        img = Image.fromarray(sr_img)
    img.save(output_img)


def generate_high_res(input_dir, output_dir):
    model = RDN(weights='noise-cancel')
    #model = RRDN(weights='gans')
    for f in sorted(os.listdir(input_dir)):
        print(f)
        generate_single_img(os.path.join(input_dir, f),
                            os.path.join(output_dir,f),
                            model)

def generate_animation(input_dir, out_file):
    pass





if __name__ == '__main__':
    input_dir = 'out/grow_small'
    out_dir = 'out/grow_big_3'
    generate_high_res(input_dir, out_dir)


