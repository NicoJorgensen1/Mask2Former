import numpy as np
import copy 
import os 
import sys 
import matplotlib 
from matplotlib import pyplot as plt 

thing_colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (185,220,255), (255,185,220),
                (220,255,185), (185,255,0), (0,185,220), (220,0,185), (115,45,115)]
class_names = ["Well", "Zona", "PV space", "Cell", "PN1", "PN2", "PN3", "PN4", "PN5", "PN6", "PN7"]

new_colors = list()
for x in list(reversed(copy.deepcopy(thing_colors))):
    t = tuple()
    for y in x:
        t += (float(y)/255,)
    new_colors.append(t + (float(1),))

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("Thing colors cmap", new_colors, len(new_colors))
boundaries = np.linspace(0, len(new_colors), len(new_colors)+1)
norm = matplotlib.colors.BoundaryNorm(boundaries, cmap.N)

n_rows = 3
n_cols = 4

fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(1+n_cols*3, n_rows*2+1))
for ax in axes.flat:
    im = ax.imshow(np.random.random((10,10)), vmin=0, vmax=1)
    ax.set_title("testing")

fig.tight_layout()
# fig.colorbar(im, ax=axes.ravel().tolist())
cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
             ax=axes.ravel().tolist(), orientation='vertical')
cbar.set_ticks(np.add(np.arange(0,len(class_names),1), 0.5))
cbar.set_ticklabels(list(reversed(copy.deepcopy(class_names))))

plt.show(block=False)




