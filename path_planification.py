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
array_data[120,120,77] = 1
array_data[150:200,100:140,50:100] = 0.5
array_data[20:220,20:220,110:155] = 0.8
array_data[10:130,10:230,5:55] = 1
array_data[60:80,10:230,75:100] = 0.7

def bresenham3D(x1, y1, z1, x2, y2, z2):
    ListOfPoints = []
    ListOfPoints.append((x1, y1, z1))
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    dz = abs(z2 - z1)
    if (x2 > x1):
        xs = 1
    else:
        xs = -1
    if (y2 > y1):
        ys = 1
    else:
        ys = -1
    if (z2 > z1):
        zs = 1
    else:
        zs = -1

    # Driving axis is X-axis"
    if (dx >= dy and dx >= dz):
        p1 = 2 * dy - dx
        p2 = 2 * dz - dx
        while (x1 != x2):
            x1 += xs
            if (p1 >= 0):
                y1 += ys
                p1 -= 2 * dx
            if (p2 >= 0):
                z1 += zs
                p2 -= 2 * dx
            p1 += 2 * dy
            p2 += 2 * dz
            ListOfPoints.append((x1, y1, z1))

    # Driving axis is Y-axis"
    elif (dy >= dx and dy >= dz):
        p1 = 2 * dx - dy
        p2 = 2 * dz - dy
        while (y1 != y2):
            y1 += ys
            if (p1 >= 0):
                x1 += xs
                p1 -= 2 * dy
            if (p2 >= 0):
                z1 += zs
                p2 -= 2 * dy
            p1 += 2 * dx
            p2 += 2 * dz
            ListOfPoints.append((x1, y1, z1))

    # Driving axis is Z-axis"
    else:
        p1 = 2 * dy - dz
        p2 = 2 * dx - dz
        while (z1 != z2):
            z1 += zs
            if (p1 >= 0):
                y1 += ys
                p1 -= 2 * dz
            if (p2 >= 0):
                x1 += xs
                p2 -= 2 * dz
            p1 += 2 * dy
            p2 += 2 * dx
            ListOfPoints.append((x1, y1, z1))
    return ListOfPoints

list_of_points = bresenham3D(120,120,77,1,1,1)
print(len(list_of_points))

score = 0
for i in range(len(list_of_points)):
    score -= array_data[list_of_points[i]]
    array_data[list_of_points[i]] = 1
print(score)

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


