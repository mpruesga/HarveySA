import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy import ndimage
import cv2
from vedo import Volume
from vedo.applications import RayCastPlotter


def image_preprocessing(image, mode):
    """
    Preprocesses input 3d image.

    Given a 3d image or 3d array, preprocesses the array depending on
    the selected mode.

    Args:
        image: a numpy 3d array.
        mode: a string determining the mode.
            'brain': creates a binary mask of the brain.
            'tumor': creates a binary mask of the tumor.
            'visualization': prepares the image for matplotlib.plt visualization.

    Returns:
        A preprocessed numpy 3d array.
    """
    processed = np.zeros(image.shape)
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
    elif mode == "visualization":
        for z in range(processed.shape[2]):
            for y in range(processed.shape[1]):
                for x in range(processed.shape[0]):
                    if image[x][y][z] < 0:
                        image[x][y][z] = 1000
    return processed


def get_tumor_dimensions(image):
    """
    Returns tumor dimensions and tumor center.

    Given a binary 3d mask of the tumor, returns the tumor dimensions
    and the tumor center represented in two arrays.

    Args:
        image: a numpy binary 3d array.

    Returns:
        tumor_dims: an array of the size of the tumor.
        tumor_center: an array of the coordinates of the tumor center.

    """
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


def modify_surface(mask):
    """
    Modifies the surface mask removing selected areas.

    Given a 3d image or array binary mask of the surface of the brain,
    removes specific selected areas, for example: forehead, brainstem
    and cerebellum.

    Args:
        mask: a binary numpy 3d.

    Returns:
        The modified mask with the specific areas removed.

    """
    y_slices = []
    for i in range(240):
        if np.sum(mask[:, i, :]) > 0:
            y_slices.append(i)

    z_slices = []
    for i in range(155):
        if np.sum(mask[:, :, i]) > 0:
            z_slices.append(i)

    face_remove = int(np.floor(len(y_slices)*0.3))
    bottom_remove = int(np.floor(len(y_slices)*0.25))

    for i in range(face_remove):
        mask[:, y_slices[i], :] = np.zeros((mask.shape[1],mask.shape[2]))
    for i in range(bottom_remove):
        mask[:, :, z_slices[i]] = np.zeros((mask.shape[0],mask.shape[1]))


def bresenham3d(x1, y1, z1, x2, y2, z2):
    """
    Finds the array of points that creates a line
    between two given points.

    Given two arrays of 3d coordinates, calculates the
    array of coordinates that are needed in order to create
    a straight line between the given points.

    Args:
        x1: the x coordinate of the first point.
        y1: the y coordinate of the first point.
        z1: the z coordinate of the first point.
        x2: the x coordinate of the second point.
        y2: the y coordinate of the second point.
        z2: the z coordinate of the second point.

    Returns:
        An array of points describing the path of a
        line between the two given coordinates.
    """
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

    # Driving axis is Z-axis
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


def sphere3d(center, d):
    """
    Finds the points needed to create a sphere of diameter d.

    Given the center of the sphere and a value d, returns the points
    needed to create a sphere of diameter d in the given center.

    Args:
        center: a numpy array of 3d coordinates.
        d: the diameter of the desired sphere.

    Returns:
        An array of points needed to create a sphere of diameter d.

    """
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
    """
    Gets the 10 best linear paths between the brain surface
    and the tumor center.

    Given a numpy 3d array image, the tumor center coordinates
    and an array of possible starts, finds the best 10 linear paths
    that minimize the total score.

    Args:
        data: a numpy 3d array or image.
        tumor_c: the coordinates of the tumor center.
        init_voxels: the possible starts for the path.

    Returns:
        A list of the best linear paths between the tumor
        center and the brain surface.

    """
    score_list = []
    for i in range(len(init_voxels)):
        voxel = init_voxels[i]
        list_of_points = bresenham3d(tumor_c[0], tumor_c[1], tumor_c[2], voxel[0], voxel[1], voxel[2])

        score = 0
        for k in range(len(list_of_points)):
            score -= data[list_of_points[k]]
            #array_data[list_of_points[k]] = 0
        score_list.append(score)
    sort_index = np.argsort(score_list)
    best_10 = sort_index[-10:]
    indexes = []
    for index in range(len(best_10)):
        voxel = init_voxels[best_10[index]]
        indexes.append(voxel)
    return indexes


def get_scores_tr(indices, data, tumor_c):
    """
    Calculates the score of a moving sphere through
    the best paths.

    Given the indices of the best linear paths, the numpy
    3d array data and the tumor center, moves a sphere through
    the path and calculates the score.

    Args:
        indices: the indices of the 10 best paths.
        data: a numpy 3d array or image.
        tumor_c: the coordinates of the tumor center.

    Returns: the list of scores for each path.

    """
    score_list = []
    path_vox = np.zeros((240, 240, 155,len(indices)))
    for i in range(len(indices)):
        idx = indices[i]
        list_of_points = bresenham3d(tumor_c[0], tumor_c[1], tumor_c[2], idx[0], idx[1], idx[2])
        score = 0
        for k in range(len(list_of_points)):
            list_of_sphere = sphere3d(list_of_points[k], 12)
            for l in range(len(list_of_sphere)):
                score -= data[list_of_sphere[l]]
                data[list_of_sphere[l]] = 0.3
                path_vox[list_of_sphere[l][0], list_of_sphere[l][1], list_of_sphere[l][2], i] = -1000
        score_list.append(score)
    return score_list, path_vox


def show_slices(data, mode):
    """
    Visualizes the 3d image.

    Given a numpy 3d array or image, creates three different
    visualization modes: sagittal, coronal and axial.
    Each mode has a slider that can move through the different
    slices.

    Args:
        data: a numpy 3d array or image

    """
    if mode == "weights":
        max = 0.3
    elif mode == "path":
        max = 800

    sagital = data[0, :, :]
    coronal = data[:, 0, :]
    axial = data[:, :, 0]

    fig, ax = plt.subplots()
    sagital_plt = plt.imshow(cv2.flip(ndimage.rotate(sagital, 90),0), cmap="gray", origin="lower", vmin=0, vmax=max)
    plt.title('Sagital', fontsize=15, fontweight='bold')
    ax1 = fig.add_axes([0.25, 0.05, 0.65, 0.03])
    slider_sagital = Slider(ax=ax1, label='Corte', valmin=0, valmax=239, valinit=0, valfmt='%d')
    ax.axis('off')
    fig, ax = plt.subplots()
    coronal_plt = plt.imshow(ndimage.rotate(coronal, 270), cmap="gray", origin="lower", vmin=0, vmax=max)
    plt.title('Coronal', fontsize=15, fontweight='bold')
    ax2 = fig.add_axes([0.25, 0.05, 0.65, 0.03])
    slider_coronal = Slider(ax2, 'Corte', 0, 239, valinit=0, valfmt='%d')
    ax.axis('off')
    fig, ax = plt.subplots()
    axial_plt = plt.imshow(ndimage.rotate(axial, 90), cmap="gray", origin="lower", vmin=0, vmax=max)
    plt.title('Axial', fontsize=15, fontweight='bold')
    ax3 = fig.add_axes([0.25, 0.05, 0.65, 0.03])
    slider_axial = Slider(ax3, 'Corte', 0, 154, valinit=0, valfmt='%d')
    ax.axis('off')





    def update(val):
        sagital_val = slider_sagital.val
        coronal_val = slider_coronal.val
        axial_val = slider_axial.val
        sagital = data[int(sagital_val), :, :]
        sagital_plt.set_data(cv2.flip(ndimage.rotate(sagital, 90),0))

        coronal = data[:, int(coronal_val), :]
        coronal_plt.set_data(ndimage.rotate(coronal, 270))

        axial = data[:,:,int(axial_val)]
        axial_plt.set_data(ndimage.rotate(axial, 90))

    slider_sagital.on_changed(update)
    slider_coronal.on_changed(update)
    slider_axial.on_changed(update)
    plt.show()


def sobel_filter(image):
    """
    Processes input image with a sobel filter.

    Given an input image or 2d array, returns the borders of the image
    utilizing a sobel filter.

    Args:
        image: a numpy 2d array.

    Returns:
        A processed numpy 2d array of the borders of an input image.

    """
    sobel_h = ndimage.sobel(image, 0, mode='constant', cval=0.1)  # horizontal gradient
    sobel_v = ndimage.sobel(image, 1, mode='constant', cval=0.1)  # vertical gradient
    magnitude = np.sqrt(sobel_h**2 + sobel_v**2)
    if np.sum(magnitude) != 0:
        magnitude *= 255.0 / np.max(magnitude)  # normalization
    return magnitude


def get_brain_surface(mask):
    """
    Returns a 3d image of the surface of the brain.

    Given a binary 3d array or image, returns the surface
    of the image.

    Args:
        mask: a binary numpy 3d array or image.

    Returns:
        A binary numpy 3d array of the surface of the mask.

    """
    kernel = np.ones((20, 20), np.uint8)
    surface = mask
    for i in range(155):
        surface[:,:,i] = cv2.morphologyEx(surface[:,:,i], cv2.MORPH_CLOSE, kernel)
        surface[:,:,i] = sobel_filter(surface[:,:,i])

    modify_surface(surface)

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


patient_id = 5

path = "MR images/Labels/WeightedSegmentation_00" + str(patient_id) + ".nii.gz"
img = nib.load(path)
img_data = img.get_fdata()


path = "MR images/Images/BraTS20_Training_00" + str(patient_id) + "_t1.nii"
og_img = nib.load(path)
og_data = og_img.get_fdata()

#show_slices(img_data, "weights")

brain_binary = image_preprocessing(og_data, "brain")
tumor_binary = image_preprocessing(img_data, "tumor")
tumor, tumor_center = get_tumor_dimensions(tumor_binary)

print(tumor)
print(tumor_center)


brain_surface, surface_index = get_brain_surface(brain_binary)

indexes = get_best_paths_s1(img_data, tumor_center, surface_index)


scores, path_3d = get_scores_tr(indexes, img_data, tumor_center)
print(scores)
print(np.argsort(scores))

new = og_data + path_3d[:, :, :, (np.argsort(scores))[-1]] + (tumor_binary*400)

image_preprocessing(new, "visualization")

show_slices(new, "path")

# Ray Caster
vol = Volume(new)
vol.mode(1).cmap("jet")
ray = RayCastPlotter(vol, bg='black', bg2='blackboard', axes=7)
ray.show(viewup="z")
ray.close()

