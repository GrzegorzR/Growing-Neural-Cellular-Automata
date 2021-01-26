import os
import pygame
import torch
import numpy as np
from PIL import Image

from lib.CAModel2 import CAModel2
from lib.displayer import displayer
from lib.utils import mat_distance
from lib.CAModel import CAModel
from lib.utils_vis import to_rgb, make_seed
from unpad import replicate_edges


def get_mask(mask_path='data/pol.jpg', map_shape=(72,72), save=False):
    im = Image.open(mask_path)
    im = im.resize(map_shape)
    im_arr = np.array(im)
    im_arr = im_arr[:,:,0] < 200
    if save:
        im = Image.fromarray(im_arr)
        im.save('tmp.png')
    return np.expand_dims(im_arr, -1)

def run_sim(model_path, save_dir=None, mask_path=None):



    eraser_radius = 15
    pix_size = 4
    map_shape = ( 120, 120)
    CHANNEL_N = 16
    CELL_FIRE_RATE = 0.2
    device = torch.device("cpu")
    if mask_path:
        mask = get_mask(mask_path, map_shape=map_shape)
    rows = np.arange(map_shape[0]).repeat(map_shape[1]).reshape([map_shape[0], map_shape[1]])
    cols = np.arange(map_shape[1]).reshape([1, -1]).repeat(map_shape[0], axis=0)
    map_pos = np.array([rows, cols]).transpose([1, 2, 0])

    map = make_seed(map_shape, CHANNEL_N)

    model = CAModel2(CHANNEL_N, CELL_FIRE_RATE, device).to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    output = model(torch.from_numpy(map.reshape([1, map_shape[0], map_shape[1], CHANNEL_N]).astype(np.float32)), 1)

    disp = displayer(map_shape, pix_size)

    isMouseDown = False
    running = True
    c = 0

    while running:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    isMouseDown = True

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    isMouseDown = False

        if isMouseDown:
            try:
                mouse_pos = np.array([int(event.pos[1]/pix_size), int(event.pos[0]/pix_size)])
                should_keep = (mat_distance(map_pos, mouse_pos)>eraser_radius).reshape([map_shape[0],map_shape[1],1])
                arr = replicate_edges(arr,{1:2,2:2})

                arr = output.detach().numpy() * should_keep #*mask

                output = torch.from_numpy(arr)
            except AttributeError:
                pass
        arr = output.detach().numpy() #* mask
        arr = replicate_edges(arr, {1: 2, 2: 2})

        output = model(output)

        map = to_rgb(output.detach().numpy()[0])
        if save_dir:
            im = Image.fromarray((map*255).astype(np.uint8))
            im.save(os.path.join(save_dir, '{}.png'.format(str(c).zfill(5))))
        c += 1
        disp.update(map)
if __name__ == '__main__':
    run_sim('models/remaster_3.pth', mask_path='data/pol.jpg', save_dir=None)