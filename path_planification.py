import numpy
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


path = "MR images/Labels/BRATS_003.nii.gz"
img = nib.load(path)
img_data = img.get_fdata()
#img_data = img_data[:,:,:,2]
print(img.header.get_data_dtype())
print(img_data.shape)


def tumor_preprocessing(image):
    for z in range(image.shape[2]):
        for y in range(image.shape[1]):
            for x in range(image.shape[0]):
                if image[x][y][z] > 1:
                    image[x][y][z] = 1
                else:
                    image[x][y][z] = 0

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


tumor_preprocessing(img_data)
tumor, tumor_center = get_tumor_dimensions(img_data)
print(tumor)
print(tumor_center)

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


def get_best_paths_s1(data):
    score_list = []
    for i in range(img_data.shape[0]):
        for j in range(img_data.shape[1]):

            list_of_points = bresenham3D(120,120,77,j,i,0)

            score = 0
            for k in range(len(list_of_points)):
                score -= data[list_of_points[k]]
                #array_data[list_of_points[k]] = 0
            score_list.append(score)

    sort_index = numpy.argsort(np.abs(score_list))
    best_10 = []
    for i in range(30):
        best_10.append(sort_index[i])
    indexes = []
    for index in range(len(best_10)):
        index_1 = best_10[index] // 240
        a = index_1 * 240
        index_2 = best_10[index] - a
        indexes.append((index_1,index_2))

    return indexes


def get_scores_tr(indexes,data):
    score_list = []
    for i in range(len(indexes)):
        idx = indexes[i]
        list_of_points = bresenham3D(120, 120, 77, idx[1], idx[0], 0)
        score = 0
        for k in range(len(list_of_points)):
            list_of_sphere = sphere3D(list_of_points[k], 10)
            for l in range(len(list_of_sphere)):
                score -= data[list_of_sphere[l]]
                data[list_of_sphere[l]] = 1
        score_list.append(score)
    return score_list


"""array_data = label_map()
indexes = get_best_paths_s1(array_data)
scores = get_scores_tr(indexes,array_data)
print(scores)"""


def show_slices(data):
    slice_0 = data[130, :, :]
    slice_1 = data[:, 80, :]
    slice_2 = data[:, :, 80]

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


show_slices(img_data)
