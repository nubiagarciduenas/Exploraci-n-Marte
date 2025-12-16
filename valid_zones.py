import copy
import numpy as np
from skimage.transform import downscale_local_mean

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

import plotly.graph_objects as px

input_file = "../Modelos/mars2.img"
output_file = "map.npy"
output_tiles_file = "tiles.npy"


data_file = open(input_file, "rb")

endHeader = False;
while not endHeader:
    line = data_file.readline().rstrip().lower()

    sep_line = line.split(b'=')
       
    if len(sep_line) == 2:
        itemName = sep_line[0].rstrip().lstrip()
        itemValue = sep_line[1].rstrip().lstrip()

        if itemName == b'valid_maximum':
            maxV = float(itemValue)
        elif itemName == b'valid_minimum':
            minV = float(itemValue)
        elif itemName == b'lines':
            n_rows = int(itemValue)
        elif itemName == b'line_samples':
            n_columns = int(itemValue)
        elif itemName == b'map_scale':
            scale_str = itemValue.split()
            if len(scale_str) > 1:
                scale = float(scale_str[0])

    elif line == b'end':
        endHeader = True
        char = 0
        while char == 0 or char == 32:
            char = data_file.read(1)[0]      
        pos = data_file.seek(-1, 1)

image_size = n_rows*n_columns
data = data_file.read(4*image_size)

image_data = np.frombuffer(data, dtype=np.dtype('f'))
image_data = image_data.reshape((n_rows, n_columns))
image_data = np.array(image_data)
image_data = image_data.astype('float64')

image_data = image_data - minV;
image_data[image_data < -10000] = -1;

sub_rate = round(10/scale) 

image_data = downscale_local_mean(image_data, (sub_rate, sub_rate))
image_data[image_data<0] = -1

print('Sub-sampling:', sub_rate)

new_scale = scale*sub_rate
print('New scale:', new_scale, 'meters/pixel')

np.save(output_file, image_data)

x = new_scale*np.arange(image_data.shape[1])
y = new_scale*np.arange(image_data.shape[0])
X, Y = np.meshgrid(x, y)

fig = px.Figure(data = px.Surface(x=X, y=Y, z=np.flipud(image_data), colorscale='hot', cmin = 0, 
                           lighting = dict(ambient = 0.0, diffuse = 0.8, fresnel = 0.02, roughness = 0.4, specular = 0.2), 
                           lightposition=dict(x=0, y=0, z=2*maxV)),
                
                layout = px.Layout(scene_aspectmode='manual', 
                                   scene_aspectratio=dict(x=1, y=image_data.shape[0]/image_data.shape[1], z=max((maxV-minV)/x.max(), 0.2)), 
                                   scene_zaxis_range = [0,maxV-minV])
                )

fig.show()

n_rows = image_data.shape[0]
n_columns = image_data.shape[1]

tile_rows = 25
tile_cols = 25

divided_map_rows = n_rows//tile_rows
divided_map_cols = n_columns//tile_cols
n_tiles = divided_map_rows*divided_map_cols

valid_tiles = np.zeros((divided_map_rows, divided_map_cols))

for i in range(divided_map_rows):
    for j in range(divided_map_cols):

        tile = image_data[i*tile_rows:(i+1)*tile_rows, j*tile_cols:(j+1)*tile_cols]

        if np.all(tile != -1) and tile.size > 0:
            tile_mean = tile.mean()
            tile_std = tile.std()

            print(tile_mean)
            valid_tiles[i,j] = tile_std < 0.6 and tile_mean < 150 and tile_mean > 100

plt.figure()
plt.imshow(valid_tiles, cmap='gray', extent =[0, new_scale*n_columns, 0, new_scale*n_rows],)
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Candidate zones')
plt.show()

np.save(output_tiles_file, valid_tiles)