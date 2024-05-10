from pathlib import Path
import numpy as np
import nibabel as nib
from skimage import io

def to_uint8(data):
    data -= data.min()
    print(data.max())
    data /= data.max()
    data *= 255
    return data.astype(np.uint8)


def nii_to_jpgs(input_path, output_dir, rgb=False):
    output_dir = Path(output_dir)
    data = nib.load(input_path).get_fdata()
    *_, num_slices, num_channels = data.shape
    for channel in range(num_channels):
        volume = data[..., channel]
        volume = to_uint8(volume)
        channel_dir = output_dir / f'channel_{channel}'
        channel_dir.mkdir(exist_ok=True, parents=True)
        for slice in range(num_slices):
            slice_data = volume[..., slice]
            if rgb:
                slice_data = np.stack(3 * [slice_data], axis=2)
            output_path = channel_dir / f'channel_{channel}_slice_{slice}.jpg'
            io.imsave(output_path, slice_data)


from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
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
#img_data[100:140,100:140,77] = 200


def show_slices(slices):
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")


slice_0 = img_data[120, :, :]
slice_1 = img_data[:, 120, :]
slice_2 = img_data[:, :, 77]
show_slices([slice_0, slice_1, slice_2])
plt.suptitle("Slices")
plt.show()
