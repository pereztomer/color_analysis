from numba import cuda, njit, float32
import time
import numpy as np
from PIL import Image, ImageFilter

# lab values according to the following study (located in color_analysis folder):
'Detecting_Diabetes_Mellitus_and_Nonproliferative_Diabetic_Retinopathy_Using_CTD.pdf'
sRGB_color_vals = np.array(
    [[188, 188, 185], [189, 99, 91], [183, 165, 180], [226, 142, 214], [136, 72, 49], [227, 150, 147],
     [225, 173, 207], [204, 183, 186], [107, 86, 56], [163, 146, 143], [200, 167, 160], [166, 129, 93]],
    dtype=np.float32)


@njit
def divide_num(number):
    if number % 2 == 0:
        return int(number / 2), int(number / 2)
    return int((number - 1) / 2), int((number - 1) / 2 + 1)


@njit
def full_image_cropping(numpy_image):
    y_nonzero, x_nonzero, _ = np.nonzero(numpy_image)  # 0 indicates black color
    y_len = np.max(y_nonzero) - np.min(y_nonzero) + 1
    x_len = np.max(x_nonzero) - np.min(x_nonzero) + 1
    max_len = max(x_len, y_len) + 5
    zero_matrix = np.zeros((max_len, max_len, 3), dtype=np.uint8)
    x_minus, x_plus = divide_num(x_len)
    y_minus, y_plus = divide_num(y_len)
    copy_numpy_image = numpy_image[int(np.min(y_nonzero)):int(np.max(y_nonzero)) + 1,
                       int(np.min(x_nonzero)):int(np.max(x_nonzero)) + 1, :].copy()

    zero_matrix[int(round(max_len / 2) - y_minus):int(round(max_len / 2) + y_plus),
    int(round(max_len / 2) - x_minus):int(round(max_len / 2) + x_plus), :] = copy_numpy_image
    numpy_image = zero_matrix
    return numpy_image


@njit
def upper_section_image_cropping(numpy_image):
    y_nonzero, x_nonzero, _ = np.nonzero(numpy_image)  # 0 indicates black color
    right_size = int(np.min(x_nonzero) + (np.max(x_nonzero) - np.min(x_nonzero)) * 1 / 3)
    left_size = int(np.min(x_nonzero) + (np.max(x_nonzero) - np.min(x_nonzero)) * 2 / 3)
    lower_side = int(np.min(y_nonzero) + (np.max(y_nonzero) - np.min(y_nonzero)) * 1 / 3)
    upper_side = np.min(y_nonzero)
    cut_numpy_image = numpy_image[upper_side:lower_side, right_size: left_size, :].copy()
    return cut_numpy_image


@cuda.jit
def return_closest_color(flat_image, output_arr):
    """
    a cuda kernel functions that finds for each pixel in the image it's closest color
    :param flat_image: a flattened np image
    :param output_arr: output array for the kernel function wo write to
    :return: return value is written into output_arr
    """
    i = cuda.grid(1)
    if i < flat_image.shape[0]:
        if flat_image[i][0] == 0 and flat_image[i][1] == 0 and flat_image[i][2] == 0:
            return
        kernel_color_vals = cuda.const.array_like(sRGB_color_vals)
        # check that that scalar listed in dist_list matches the amount of colors to compare too
        dist_list = cuda.local.array(shape=12, dtype=float32)
        for index in range(len(kernel_color_vals)):
            temp_dist = 0
            for arr_index in range(3):
                temp_dist += (flat_image[i][arr_index] - kernel_color_vals[index][arr_index]) ** 2
            dist = temp_dist ** 0.5
            dist_list[index] = dist

        min_index = 0
        min_val = 100000.0
        for dist_index in range(12):
            if dist_list[dist_index] < min_val:
                min_val = dist_list[dist_index]
                min_index = dist_index
        output_arr[i] = min_index


@njit
def calc_image_feature_vec_cuda(output_arr, feature_vec):
    """
    calculates the percentage of each of the colors in output array
    :param output_arr: a numpy array containing colors
    :param feature_vec: empty feature vector
    :return: a feature vector
    """
    non_zero_pixels_counter = 0
    for relevant_color_index in output_arr:
        if relevant_color_index == -1:  # -1 marks that the location was not processed
            continue
        non_zero_pixels_counter += 1
        feature_vec[int(relevant_color_index)] += 1

    feature_vec = feature_vec / non_zero_pixels_counter
    return feature_vec


@njit
def calc_image_feature_vec_sum_cuda(output_arr, feature_vec):
    """
    calculates the percentage of each of the colors in output array
    :param output_arr: a numpy array containing colors
    :param feature_vec: empty feature vector
    :return: a feature vector
    """
    non_zero_pixels_counter = 0
    for relevant_color_index in output_arr:
        if relevant_color_index == -1:  # -1 marks that the location was not processed
            continue
        non_zero_pixels_counter += 1
        feature_vec[int(relevant_color_index)] += np.sum(sRGB_color_vals[int(relevant_color_index)])

    feature_vec = feature_vec / non_zero_pixels_counter
    return feature_vec


def calc_image_feature_vec_cuda_wrapper(address: str, gaussian_blur=False, image_cut='full_image') -> np.array:
    """
    for a given image located in address var calculates color feature vector
    :param address: string address to the specific image
    :param gaussian_blur: a flag indicating if to apply gaussian_blur to the image
    :param image_cut: full_image / upper_section of the image
    :return: a feature vector
    """
    im = Image.open(address)
    if gaussian_blur:
        im = im.filter(ImageFilter.GaussianBlur(radius=9))
    np_image = np.asarray(im, dtype=np.float32)
    if image_cut == 'full_image':
        np_image = full_image_cropping(np_image)
    elif image_cut == 'upper_section':
        np_image = upper_section_image_cropping(np_image)
    else:
        raise Exception('You can only pass for variable "image_cut": full_image / upper_section')

    flat_image = np.reshape(np_image, newshape=(np_image.shape[0] * np_image.shape[1], np_image.shape[2]))
    output_arr = np.zeros(shape=flat_image.shape[0])

    flat_image = cuda.to_device(flat_image)
    output_arr = cuda.to_device(output_arr)

    threadsperblock = 32

    # Calculate the number of thread blocks in the grid
    blockspergrid = (np_image.size + (threadsperblock - 1)) // threadsperblock
    return_closest_color[blockspergrid, threadsperblock](flat_image, output_arr)
    output_arr = output_arr.copy_to_host()

    feature_vec = np.zeros(shape=sRGB_color_vals.shape[0], dtype=np.float32)
    feature_vec = calc_image_feature_vec_sum_cuda(output_arr, feature_vec)
    return feature_vec


def main():
    folder_address = '/home/beast/GitProjects/DeepLearning/full_db_flavor_C'

    images_lst = ['C2360.png', 'C2595.png', 'C2642.png', 'C2804.png', 'C2824.png', 'C3111.png', 'C2358.png',
                  'C3436.png', 'C3683.png', 'C4094.png']
    init_time = time.time()
    for image in images_lst:
        feature_vec = calc_image_feature_vec_cuda_wrapper(f'{folder_address}/{image}', gaussian_blur=True)

    total_time = time.time() - init_time
    print(f'total time: {total_time}, average time: {total_time / len(images_lst)}')


if __name__ == '__main__':
    main()
