import glob
import os
import tensorflow as tf
from Code.color_analysis.utilities.custom_utils import generate_dict_according_to_query
from PIL import Image
import numpy as np
import json
import time
import tqdm
import random
from sklearn.cluster import KMeans

MB = 1000000
MAX_BYTES = 200 * MB


def parse_single_image(image):
    # define the dictionary -- the structure -- of our single example
    data = {
        'raw_image': _bytes_feature(serialize_array(image)),
    }
    # create an Example, wrapping the single features
    out = tf.train.Example(features=tf.train.Features(feature=data))
    return out


def serialize_array(array):
    array = tf.io.serialize_tensor(array)
    return array


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):  # if value ist tensor
        value = value.numpy()  # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def image_cropping(numpy_image):
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


def divide_num(number):
    if number % 2 == 0:
        return int(number / 2), int(number / 2)
    return int((number - 1) / 2), int((number - 1) / 2 + 1)


def generate_relevant_images_ids(ds_address, query, photos_num):
    random.seed(100)

    total_db_addresses = glob.glob(f'{ds_address}/*.png')
    total_db_addresses = [x.split('/')[-1].replace('.png', '').replace('C', '') for x in total_db_addresses]
    patient_id_to_photo_id_dict = generate_dict_according_to_query(query=query,
                                                                   available_images=total_db_addresses,
                                                                   label=1)
    # choose randomly the image id for a given patient
    patient_ids = []
    for key, value in patient_id_to_photo_id_dict.items():
        chosen_value = random.choice(value)
        patient_ids.append(chosen_value[0])

    patient_ids = random.sample(patient_ids, photos_num)
    # current_run_path = f'{destination_path}/{current_run_name}'

    # os.makedirs(f'{current_run_path}/tf_records', exist_ok=True)
    #
    # stats = {}
    # stats['set'] = current_run_name
    # stats['query_polyp'] = query
    # stats['polyp_patient_id'] = list(patient_ids)
    # stats['num_polyp_patient_id'] = len(patient_ids)
    # stats['flavor'] = 'C'
    # with open(f'{current_run_path}/stats_points_for_kmeans.json', "w") as outfile:
    #     outfile.write(json.dumps(stats, indent=4))
    counter = 0
    lab_pixels_list = []
    counter_writer = 0
    current_byte_sum = 0
    # tf_writer = tf.io.TFRecordWriter(f'{current_run_path}/tf_records/k_means_tf_records_{counter_writer}')
    pbar = tqdm.tqdm(total=len(patient_ids))
    for image_id in patient_ids:
        image_path = f'{ds_address}/C{image_id}.png'
        im = Image.open(image_path)
        np_image = np.asarray(im, dtype=np.float32)
        np_image = image_cropping(np_image)
        flat_image = np.reshape(np_image, newshape=(np_image.shape[0] * np_image.shape[1], np_image.shape[2]))
        for pixel in flat_image:
            if pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0:
                continue
            lab_pixel = convert_srgb_lab(pixel)
            lab_pixels_list.append(lab_pixel)
            # np.expand_dims(kmeans.cluster_centers_, axis=1).astype(np.uint8)
            # out = parse_single_image(lab_pixel.astype(np.float32))
            # current_byte_sum += out.ByteSize()
            # if current_byte_sum > MAX_BYTES:
            #     tf_writer.close()
            #     counter_writer += 1
            #     tf_writer = tf.io.TFRecordWriter(f'{current_run_path}/tf_records/k_means_tf_records_{counter_writer}')
            #     current_byte_sum = 0
            #
            # tf_writer.write(out.SerializeToString())
        pbar.update(1)
    # tf_writer.close()
    pbar.close()
    return lab_pixels_list


def f(val):
    k = 903.3  # 24389 / 27
    epsilon = 0.008856  # 216 / 24389
    if val > epsilon:
        return val ** (1 / 3)
    return (k * val + 16) / 116


def convert_srgb_lab(pixel_rgb):
    expand_srgb = expand_rgb(pixel_rgb)
    srgb2xyz = np.array([[0.4124530, 0.3575800, 0.180423],
                         [0.212671, 0.715160, 0.072169],
                         [0.019334, 0.119193, 0.950227]])

    xyz_pixel = srgb2xyz @ expand_srgb

    x_func_out = f(xyz_pixel[0] / 0.95044921)
    y_func_out = f(xyz_pixel[1])
    z_func_out = f(xyz_pixel[2] / 1.0888)
    Lab_pixel = np.zeros(3)
    Lab_pixel[0] = L = 116 * y_func_out - 16
    Lab_pixel[1] = a = 500 * (x_func_out - y_func_out)
    Lab_pixel[2] = b = 200 * (y_func_out - z_func_out)
    return Lab_pixel


def expand_rgb(pixel_rgb):
    expanded_srgb = np.zeros(3)
    # conversion function for rgb non-linear transformation
    for idx in range(3):
        if pixel_rgb[idx] <= 10:
            expanded_srgb[idx] = pixel_rgb[idx] / 3294.6
        else:
            expanded_srgb[idx] = ((pixel_rgb[idx] + 14.025) / 269.025) ** 2.4
    return expanded_srgb


def main():
    current_run_name = 'Lab_full_image_noBlur'
    number_photos = 200
    centroids_list = [25, 50]
    full_ds_address = '/home/beast/GitProjects/DeepLearning/full_db_flavor_C'
    destination_path = '/home/beast/GitProjects/DeepLearning/data/k_means'
    query_polyp = "select id, patientid from gixammain.dbo.ImageStudyPatient where deleted = 0 and rejected = 0" \
                  " and C_PolypATLG1 = 1 and [position] = 0 and approved=1" \
                  " and locked = 1 and branchid in (4,6,8)"
    query_healthy = f'select id, patientid from gixammain.dbo.ImageStudyPatient where deleted = 0 ' \
                    f'and rejected = 0 and approved=1 and colonoscopy=1 and C_CRC=0 and C_PolypATLG1=0 and ' \
                    f'C_PolypATGT1=0 and C_PolypSLT1=0 and C_PolypSGT1=0 and C_PolypHyperPlasty=0 ' \
                    f'and C_Colitis=0 and [position] = 0 and locked = 1 and branchid in (4,6,8)'
    query_dict = {'polyp': query_polyp, 'healthy': query_healthy}

    images_list = []
    for query_type, query in query_dict.items():
        images_list.extend(generate_relevant_images_ids(full_ds_address, query, number_photos//2))

    for k_value in centroids_list:
        kmeans = KMeans(n_clusters=k_value, random_state=0, init='k-means++').fit(images_list)
        os.makedirs(f'{destination_path}/{current_run_name}/healthy_and_polyp', exist_ok=True)
        with open(
                f'{destination_path}/{current_run_name}/healthy_and_polyp/'
                f'{number_photos}_pics_stats_points_for_kmeans_k_{k_value}.json',
                "w") as outfile:
            outfile.write(json.dumps(kmeans.cluster_centers_.tolist(), indent=4))


if __name__ == '__main__':
    main()

    # f = open('/home/tomer/k_means/Lab_polyp/100_pics_stats_points_for_kmeans_k_12.json')
    # file_100 = json.load(f)
    #
    # original_file = file_100
    # file_100_norm = []
    # for idx, pixel_center in enumerate(file_100):
    #     file_100_norm.append((idx, np.linalg.norm(pixel_center)))
    #
    # sorted_file_100_norm = sorted(file_100_norm, key=lambda x: x[1])
    #
    # sorted_file_100_norm = [x[0] for x in sorted_file_100_norm]
    # file_100 = np.array(file_100)
    # sorted_file_100 = file_100[sorted_file_100_norm]
    #
    # f = open('/home/tomer/k_means/Lab_polyp/250_pics_stats_points_for_kmeans_k_12.json')
    # file_250 = np.array(json.load(f))
    #
    # file_250_norm = []
    # for idx, pixel_center in enumerate(file_250):
    #     file_250_norm.append((idx, np.linalg.norm(pixel_center)))
    #
    # sorted_file_100_norm = sorted(file_250_norm, key=lambda x: x[1])

    # f = open('/home/tomer/k_means/Lab_polyp/100_pics_stats_points_for_kmeans_k_12.json')
    # file_100 = json.load(f)
    #
    # original_file = file_100
    # file_100_norm = []
    # for idx, pixel_center in enumerate(file_100):
    #     file_100_norm.append((idx, np.linalg.norm(pixel_center)))
    #
    # sorted_file_100_norm = sorted(file_100_norm, key=lambda x: x[1])
    #
    # sorted_file_100_norm = [x[0] for x in sorted_file_100_norm]
    # file_100 = np.array(file_100)
    # sorted_file_100 = file_100[sorted_file_100_norm]
    #
    # f = open('/home/tomer/k_means/Lab_polyp/250_pics_stats_points_for_kmeans_k_12.json')
    # file_250 = np.array(json.load(f))
    #
    # file_250_norm = []
    # for idx, pixel_center in enumerate(file_250):
    #     file_250_norm.append((idx, np.linalg.norm(pixel_center)))
    #
    # sorted_file_100_norm = sorted(file_250_norm, key=lambda x: x[1])
    #
    # sorted_file_100_norm = [x[0] for x in sorted_file_100_norm]
    # file_250 = np.array(file_250)
    # sorted_file_250 = file_250[sorted_file_100_norm]
    #
    # print(np.linalg.norm(sorted_file_250-sorted_file_100))
