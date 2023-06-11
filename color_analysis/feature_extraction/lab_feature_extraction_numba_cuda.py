from numba import cuda, njit, float32
import time
import numpy as np
from PIL import Image, ImageFilter

# lab values according to the following study (located in color_analysis folder):
'Detecting_Diabetes_Mellitus_and_Nonproliferative_Diabetic_Retinopathy_Using_CTD.pdf'
# lab_color_vals = np.array([[76.0693, -0.5580, 1.3615],
#                            [52.2540, 34.8412, 21.3002],
#                            [69.4695, 9.5423, -5.4951],
#                            [69.4695, 42.4732, -23.8880],
#                            [37.8424, 24.5503, 25.9396],
#                            [69.4695, 28.4947, 13.3940],
#                            [76.0693, 24.3246, -9.7749],
#                            [76.0693, 7.8917, 0.9885],
#                            [37.8424, 3.9632, 20.5874],
#                            [61.6542, 5.7160, 3.7317],
#                            [70.9763, 10.9843, 8.2952],
#                            [56.3164, 9.5539, 24.4546]], dtype=np.float32)


# sRGB to cie xyz conversion matrix
srgb2xyz = np.array([[0.4124530, 0.3575800, 0.180423],
                     [0.212671, 0.715160, 0.072169],
                     [0.019334, 0.119193, 0.950227]])


@njit
def divide_num(number):
    if number % 2 == 0:
        return int(number / 2), int(number / 2)
    return int((number - 1) / 2), int((number - 1) / 2 + 1)


@njit
def full_image_cropping(numpy_image):
    """
     crop from a given image it's non-black parts
     :param numpy_image: numpy array image
     :return: numpy array image
     """
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
    """
     crop from a given image it's upper part
     :param numpy_image: numpy array image
     :return: numpy array image
     """
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
        kernel_color_vals = cuda.const.array_like(lab_color_vals)  # lab_color_vals
        kernel_srgb2xyz = cuda.const.array_like(srgb2xyz)  # srgb2xyz
        # check that that scalar listed in dist_list matches the amount of colors to compare too
        dist_list = cuda.local.array(shape=centroids_num,
                                     dtype=float32)  # np.zeros(shape=lab_color_vals.shape[0], dtype=np.float32) #
        xyz = cuda.local.array(shape=3, dtype=float32)  # np.zeros(shape=3, dtype=np.float32)

        expanded_srgb = cuda.local.array(shape=3, dtype=float32)  # np.zeros(shape=3, dtype=np.float32)
        # conversion function for rgb non-linear transformation
        for idx in range(3):
            if flat_image[i][idx] <= 10:
                expanded_srgb[idx] = flat_image[i][idx] / 3294.6
            else:
                expanded_srgb[idx] = ((flat_image[i][idx] + 14.025) / 269.025) ** 2.4

        xyz[0] = 0
        xyz[1] = 0
        xyz[2] = 0
        for index in range(3):
            xyz[0] += expanded_srgb[index] * kernel_srgb2xyz[0][index]
            xyz[1] += expanded_srgb[index] * kernel_srgb2xyz[1][index]
            xyz[2] += expanded_srgb[index] * kernel_srgb2xyz[2][index]

        # the calculation below convert from cie xyz color space to Lab color space
        k = 24389 / 27
        epsilon = 216 / 24389

        if (xyz[0] / 0.95044921) > epsilon:
            x_func_out = (xyz[0] / 0.95044921) ** (1 / 3)
        else:
            x_func_out = (k * (xyz[0] / 0.95044921) + 16) / 116

        if xyz[1] > epsilon:
            y_func_out = xyz[1] ** (1 / 3)
        else:
            y_func_out = (k * xyz[1] + 16) / 116

        if (xyz[2] / 1.0888) > epsilon:
            z_func_out = (xyz[2] / 1.0888) ** (1 / 3)
        else:
            z_func_out = (k * (xyz[2] / 1.0888) + 16) / 116

        L = 116 * y_func_out - 16
        a = 500 * (x_func_out - y_func_out)
        b = 200 * (y_func_out - z_func_out)

        pixel_lab_vals = cuda.local.array(shape=3, dtype=float32)  # np.zeros(shape=3, dtype=np.float32)
        pixel_lab_vals[0] = L
        pixel_lab_vals[1] = a
        pixel_lab_vals[2] = b

        for index in range(len(kernel_color_vals)):
            temp_dist = 0
            for arr_index in range(3):
                temp_dist += (pixel_lab_vals[arr_index] - kernel_color_vals[index][arr_index]) ** 2
            dist = temp_dist ** 0.5
            dist_list[index] = dist

        min_index = 0
        min_val = 100000.0
        for dist_index in range(lab_color_vals.shape[0]):
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


# @njit
# def calc_image_feature_vec_sum_cuda(output_arr, feature_vec):
#     """
#     calculates the percentage of each of the colors in output array
#     :param output_arr: a numpy array containing colors
#     :param feature_vec: empty feature vector
#     :return: a feature vector
#     """
#     non_zero_pixels_counter = 0
#     for relevant_color_index in output_arr:
#         if relevant_color_index == -1:  # -1 marks that the location was not processed
#             continue
#         non_zero_pixels_counter += 1
#         feature_vec[int(relevant_color_index)] += np.sum(lab_color_vals[int(relevant_color_index)])
#
#     feature_vec = feature_vec / non_zero_pixels_counter
#     return feature_vec


def calc_image_feature_vec_cuda_wrapper(address: str, centroids, gaussian_blur=False,
                                        image_cut='full_image') -> np.array:
    """
    for a given image located in address var calculates color feature vector
    :param address: string address to the specific image
    :param centroids: centroids to calculates feature vector from
    :param gaussian_blur: a flag indicating if to apply gaussian_blur to the image
    :param image_cut: full_image / upper_section of the image
    :return: a feature vector
    """
    global lab_color_vals
    lab_color_vals = centroids
    global centroids_num
    centroids_num = lab_color_vals.shape[0]
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
    output_arr = np.ones(shape=flat_image.shape[0]) * -1  # creating an array were each element is -1

    flat_image = cuda.to_device(flat_image)
    output_arr = cuda.to_device(output_arr)

    threadsperblock = 32

    # Calculate the number of thread blocks in the grid
    blockspergrid = (np_image.size + (threadsperblock - 1)) // threadsperblock
    return_closest_color[blockspergrid, threadsperblock](flat_image, output_arr)
    output_arr = output_arr.copy_to_host()

    feature_vec = np.zeros(shape=lab_color_vals.shape[0], dtype=np.float32)

    feature_vec = calc_image_feature_vec_cuda(output_arr, feature_vec)
    assert np.linalg.norm(feature_vec) > 0
    return feature_vec


def main():
    folder_address = '/home/beast/GitProjects/DeepLearning/full_db_flavor_C'

    images_lst = ['C2360.png', 'C2595.png', 'C2642.png', 'C2804.png', 'C2824.png', 'C3111.png', 'C2358.png',
                  'C3436.png', 'C3683.png', 'C4094.png']
    init_time = time.time()
    for image in images_lst:
        try:
            feature_vec = calc_image_feature_vec_cuda_wrapper(f'{folder_address}/{image}', gaussian_blur=False)
        except Exception as e:
            print(image)
            print(e)
    total_time = time.time() - init_time
    print(f'total time: {total_time}, average time: {total_time / len(images_lst)}')


if __name__ == '__main__':
    main()
