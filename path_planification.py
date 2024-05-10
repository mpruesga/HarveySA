from pathlib import Path
import numpy as np
import nibabel as nib
from skimage import io
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from skimage import measure


def plot_3d(image, threshold=-300):
    p = image.transpose(2,1,0)
    verts, faces, normals, values = measure.marching_cubes_lewiner(p, threshold)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)
    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()


path = "MR images/BRATS_486.nii.gz"
img = nib.load(path)
img_data = img.get_fdata()
img_data = img_data[:,:,:,2]
print(img.header.get_data_dtype())
print(img_data.shape)


array_data = np.zeros([240,240,155])
array_data[100:140,100:140,75:100] = 1
array_data[150:200,100:140,50:100] = 0.5
array_data[20:220,20:220,110:155] = 0.8
array_data[10:130,10:230,40:55] = 0.8
array_data[60:80,10:230,75:100] = 0.7

def show_slices(slices):
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")


slice_0 = array_data[1, :, :]
slice_1 = array_data[:, 2, :]
slice_2 = array_data[:, :, 80]

slice = plt.imshow(slice_2, cmap="gray", origin="lower")

axidx = plt.axes([0.25, 0.15, 0.65, 0.03])
slidx = Slider(axidx, 'index', 0, 154, valinit=0, valfmt='%d')

def update(val):
    idx = slidx.val
    slice_2 = array_data[:,:,int(idx)]
    slice.set_data(slice_2)


slidx.on_changed(update)

plt.show()

