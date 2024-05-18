import numpy
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy import ndimage
import cv2


path = "MR images/Labels/WeightedSegmentation_001_Test.nii.gz"
img = nib.load(path)
img_data = img.get_fdata()
#img_data = img_data[:,:,:,2]
print(img.header.get_data_dtype())
print(img_data.shape)


def image_preprocessing(image, mode):
    processed = np.zeros([240, 240, 155])
    if mode == "brain":
        for z in range(processed.shape[2]):
            for y in range(processed.shape[1]):
                for x in range(processed.shape[0]):
                    if image[x][y][z] != 0:
                        processed[x][y][z] = 1
                    else:
                        processed[x][y][z] = 0
    elif mode == "tumor":
        for z in range(processed.shape[2]):
            for y in range(processed.shape[1]):
                for x in range(processed.shape[0]):
                    if image[x][y][z] < 0:
                        processed[x][y][z] = 1
                    else:
                        processed[x][y][z] = 0
    elif mode == "viz":
        for z in range(processed.shape[2]):
            for y in range(processed.shape[1]):
                for x in range(processed.shape[0]):
                    if image[x][y][z] < 0:
                        image[x][y][z] = 0
                    if image[x][y][z] == 1:
                        image[x][y][z] = 0.3
    return processed


def get_tumor_dimensions(image):
    x_slices = []
    for i in range(240):
        if np.sum(image[i,:,:]) > 0:
            x_slices.append(i)
    y_slices = []
    for j in range(240):
        if np.sum(image[:, j, :]) > 0:
            y_slices.append(j)
    z_slices = []
    for k in range(155):
        if np.sum(image[:,:,k]) > 0:
            z_slices.append(k)

    tumor_dims = (max(x_slices)-min(x_slices),max(y_slices)-min(y_slices),max(z_slices)-min(z_slices))

    if len(x_slices) % 2 != 0:  # Check if the length is odd
        middle_index = len(x_slices) // 2
    else:
        middle_index = len(x_slices) // 2 - 1
    x_center = x_slices[middle_index]

    if len(y_slices) % 2 != 0:  # Check if the length is odd
        middle_index = len(y_slices) // 2
    else:
        middle_index = len(y_slices) // 2 - 1
    y_center = y_slices[middle_index]

    if len(z_slices) % 2 != 0:  # Check if the length is odd
        middle_index = len(z_slices) // 2
    else:
        middle_index = len(z_slices) // 2 - 1
    z_center = z_slices[middle_index]

    tumor_center = (x_center,y_center,z_center)
    return tumor_dims, tumor_center


def get_brain_midline(brain_binary, img):
    x_slices = []
    for i in range(240):
        if np.sum(brain_binary[i, :, :]) > 0:
            x_slices.append(i)

    if len(x_slices) % 2 != 0:  # Check if the length is odd
        middle_index = len(x_slices) // 2
    else:
        middle_index = len(x_slices) // 2 - 1
    x_center = x_slices[middle_index]

    for y in range(img.shape[0]):
        for z in range(img.shape[2]):
            if brain_binary[x_center,y,z] == 1:
                img_data[x_center,y,z] = 1
                img_data[x_center+1, y, z] = 1
                img_data[x_center-1, y, z] = 1



def label_map():
    array_data = np.zeros([240,240,155])
    array_data[120,120,77] = -1
    array_data[150:200,100:140,50:100] = 0.5
    array_data[20:220,20:220,110:155] = 0.8
    array_data[10:130,10:230,5:55] = 1
    array_data[60:80,10:230,75:100] = 0.7
    return array_data


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


def sphere3D(center,d):
    list_of_points_sphere = []
    start = []
    for i in range(len(center)):
        start.append(center[i] - d//2)
    for z in range(d):
        for y in range(d):
            for x in range(d):
                if ((start[0]+x < 240) & (start[1]+y < 240) & (start[2]+z < 155)) & ((start[0]+x >= 0) & (start[1]+y >= 0) & (start[2]+z >= 0)):
                    dist_to_center = np.sqrt((center[0]-(start[0]+x))**2+(center[1]-(start[1]+y))**2+(center[2]-(start[2]+z))**2)
                    if dist_to_center <= d/2:
                        list_of_points_sphere.append((start[0]+x, start[1]+y, start[2]+z))
    return list_of_points_sphere


def get_best_paths_s1(data, tumor_c, init_voxels):
    score_list = []
    for i in range(len(init_voxels)):
        voxel = init_voxels[i]
        list_of_points = bresenham3D(tumor_c[0],tumor_c[1],tumor_c[2],voxel[0],voxel[1],voxel[2])

        score = 0
        for k in range(len(list_of_points)):
            score -= data[list_of_points[k]]
            #array_data[list_of_points[k]] = 0
        score_list.append(score)

    sort_index = numpy.argsort(np.abs(score_list))
    best_10 = []
    for i in range(10):
        best_10.append(sort_index[i])
    indexes = []
    for index in range(len(best_10)):
        voxel = init_voxels[best_10[index]]
        indexes.append(voxel)
        """index_1 = best_10[index] // 240
        a = index_1 * 240
        index_2 = best_10[index] - a
        indexes.append((index_1,index_2))"""

    return indexes


def get_scores_tr(indexes,data,tumor_c):
    score_list = []
    for i in range(len(indexes)):
        idx = indexes[i]
        list_of_points = bresenham3D(tumor_c[0],tumor_c[1],tumor_c[2], idx[0], idx[1], idx[2])
        score = 0
        for k in range(len(list_of_points)):
            list_of_sphere = sphere3D(list_of_points[k], 5)
            for l in range(len(list_of_sphere)):
                score -= data[list_of_sphere[l]]
                data[list_of_sphere[l]] = 0.3
        score_list.append(score)
    return score_list


def show_slices(data):
    slice_0 = data[100, :, :]
    slice_1 = data[:, 140, :]
    slice_2 = data[:, :, 66]

    fig, axes = plt.subplots(1,3)
    slice0 = axes[0].imshow(slice_0, cmap="gray", origin="lower")
    slice1 = axes[1].imshow(slice_1, cmap="gray", origin="lower")
    slice2 = axes[2].imshow(slice_2, cmap="gray", origin="lower")

    axidx = plt.axes([0.25, 0.15, 0.65, 0.03])
    slidx = Slider(axidx, 'index', 0, 239, valinit=0, valfmt='%d')

    def update(val):
        idx = slidx.val
        slice_0 = data[int(idx), :, :]
        slice0.set_data(slice_0)

        slice_1 = data[:, int(idx), :]
        slice1.set_data(slice_1)

        if idx < 155:
            slice_2 = data[:,:,int(idx)]
            slice2.set_data(slice_2)

    slidx.on_changed(update)
    plt.show()


def sobel_filter(image):
    sobel_h = ndimage.sobel(image, 0, mode='constant', cval=0.1)  # horizontal gradient
    sobel_v = ndimage.sobel(image, 1, mode='constant', cval=0.1)  # vertical gradient
    magnitude = np.sqrt(sobel_h**2 + sobel_v**2)
    if np.sum(magnitude) != 0:
        magnitude *= 255.0 / np.max(magnitude)  # normalization
    """fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    plt.gray()  # show the filtered result in grayscale
    axs[0, 0].imshow(image)
    axs[0, 1].imshow(sobel_h)
    axs[1, 0].imshow(sobel_v)
    axs[1, 1].imshow(magnitude)
    titles = ["original", "horizontal", "vertical", "magnitude"]
    for i, ax in enumerate(axs.ravel()):
        ax.set_title(titles[i])
        ax.axis("off")
    #plt.show()"""
    return magnitude


def get_brain_surface(mask):
    kernel = np.ones((20, 20), np.uint8)
    surface = mask
    for i in range(155):
        surface[:,:,i] = cv2.morphologyEx(surface[:,:,i], cv2.MORPH_CLOSE, kernel)
        surface[:,:,i] = sobel_filter(surface[:,:,i])

    surface_indexes = []
    for z in range(surface.shape[2]):
        for y in range(surface.shape[1]):
            for x in range(surface.shape[0]):
                if surface[x,y,z] > 200:
                    surface[x,y,z] = 1
                    surface_indexes.append((x,y,z))
                else:
                    surface[x,y,z] = 0
    return surface, surface_indexes


brain_binary = image_preprocessing(img_data,"brain")
tumor_binary = image_preprocessing(img_data,"tumor")
tumor, tumor_center = get_tumor_dimensions(tumor_binary)
get_brain_midline(brain_binary, img_data)

print(tumor)
print(tumor_center)


brain_surface, surface_index = get_brain_surface(brain_binary)

indexes = get_best_paths_s1(img_data, tumor_center, surface_index)
print(indexes)
scores = get_scores_tr(indexes, img_data, tumor_center)
print(scores)
image_preprocessing(img_data, "viz")
show_slices(img_data)