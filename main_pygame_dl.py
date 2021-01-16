import pygame
import torch
import numpy as np
from PIL import Image

from lib.displayer import displayer
from lib.utils import mat_distance
from lib.CAModel import CAModel
from lib.utils_vis import to_rgb, make_seed

eraser_radius = 10
pix_size = 8
_map_shape = (72,72)
CHANNEL_N = 16
CELL_FIRE_RATE = 0.5
model_path = "models/growing.pth"
device = torch.device("cpu")

_rows = np.arange(_map_shape[0]).repeat(_map_shape[1]).reshape([_map_shape[0],_map_shape[1]])
_cols = np.arange(_map_shape[1]).reshape([1,-1]).repeat(_map_shape[0],axis=0)
_map_pos = np.array([_rows,_cols]).transpose([1,2,0])

_map = make_seed(_map_shape, CHANNEL_N)

model = CAModel(CHANNEL_N, CELL_FIRE_RATE, device).to(device)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
output = model(torch.from_numpy(_map.reshape([1,_map_shape[0],_map_shape[1],CHANNEL_N]).astype(np.float32)), 1)

disp = displayer(_map_shape, pix_size)

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
            should_keep = (mat_distance(_map_pos, mouse_pos)>eraser_radius).reshape([_map_shape[0],_map_shape[1],1])
            arr = output.detach().numpy() * should_keep
            arr = np.pad(arr, (10, 10), 'linear_ramp',
                 end_values=(0, 0))
            output = torch.from_numpy(output.detach().numpy()*should_keep)
        except AttributeError:
            pass

    output = model(output, 1)

    _map = to_rgb(output.detach().numpy()[0])
    im = Image.fromarray((_map*255).astype(np.uint8))
    im.save('out/grow_small/{}.png'.format(str(c).zfill(5)))
    c += 1
    disp.update(_map)
