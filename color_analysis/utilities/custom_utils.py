import glob
import json
import numpy as np
import pyodbc

CONNECTION_STRING = 'Driver={ODBC Driver 18 for SQL Server};Server=tcp:gixamsqlserver.database.windows.net,1433;Database=gixammain;Persist Security Info=False;UID=admindb;PWD=ItsG00dT0BeH0me;MultipleActiveResultSets=False;Connection Timeout=80;'


def remove_images(quality_measures_address: str,
                  sharpness_threshold: float,
                  num_of_pixel_std_deviations: float,
                  num_of_reflection_std_deviations: float):
    """
    removing images according to measures: connected components, sharpness, percentage of non-black pixels and reflection
    :param quality_measures_address: folder of the json file containing scores for each image for each statistic
    :param sharpness_threshold: threshold filtering image according to its sharpness score
    :param num_of_pixel_std_deviations: number of std deviations for pixel count
    :param num_of_reflection_std_deviations: number of std deviations reflection
    :return:
    """

    removed_photos_connected_components = set()
    f = open(f'{quality_measures_address}/components_dict.json')
    connected_components_data = json.load(f)

    for photo_address, num_connected_components in connected_components_data:
        photo_id = photo_address.split('/')[-1].replace('.png', '')
        if num_connected_components != 2:
            removed_photos_connected_components.add(photo_id)

    removed_photos_sharpness = set()
    f = open(f'{quality_measures_address}/sharpness_dict.json')
    sharpness_data = json.load(f)

    for photo_address, sharpness_score in sharpness_data:
        photo_id = photo_address.split('/')[-1].replace('.png', '')
        if sharpness_score < sharpness_threshold:
            removed_photos_sharpness.add(photo_id)

    removed_photos_too_little_pixels = set()
    f = open(f'{quality_measures_address}/percentage_dict.json')
    pixels_data = json.load(f)
    pixel_percentage_lst = []
    photo_id_set = set()
    for photo_address, pixels_percentage in pixels_data:
        photo_id = photo_address.split('/')[-1].replace('.png', '')
        if photo_id not in photo_id_set:
            photo_id_set.add(photo_id)
            pixel_percentage_lst.append(pixels_percentage)

    pixel_mean = np.mean(pixel_percentage_lst, axis=0)
    pixel_std = np.std(pixel_percentage_lst, axis=0)

    for photo_address, pixels_percentage in pixels_data:
        photo_id = photo_address.split('/')[-1].replace('.png', '')
        if pixels_percentage < pixel_mean - pixel_std * num_of_pixel_std_deviations:
            removed_photos_too_little_pixels.add(photo_id)

    total_reflection_photos = set()
    removed_reflection_photos = set()
    reflection_lst = []
    f = open(f'{quality_measures_address}/reflection_dict.json')
    reflection_data = json.load(f)

    for photo_address, reflection_vals in reflection_data.items():
        red_reflection_val = reflection_vals[1]
        photo_id = photo_address.split('/')[-1].replace('.png', '')
        if photo_id not in total_reflection_photos:
            total_reflection_photos.add(photo_id)
            reflection_lst.append(red_reflection_val)

    reflection_mean = np.mean(reflection_lst, axis=0)
    reflection_std = np.std(reflection_lst, axis=0)

    for photo_address, reflection_vals in reflection_data.items():
        photo_id = photo_address.split('/')[-1].replace('.png', '')
        if reflection_vals[1] > reflection_mean + reflection_std * num_of_reflection_std_deviations:
            removed_reflection_photos.add(photo_id)

    total_removed_photos = set()
    total_removed_photos.update(removed_photos_connected_components)
    total_removed_photos.update(removed_photos_sharpness)
    total_removed_photos.update(removed_reflection_photos)
    return total_removed_photos


def find_existing_images(address, flavor):
    existing_images = []
    for address in glob.glob(f'{address}/*.png'):
        photo_id = address.split('/')[-1].replace('.png', '')
        if flavor in photo_id:
            photo_id = photo_id.replace(flavor, '')
            existing_images.append(photo_id)

    return existing_images


def chose_flavor(flavor, images_to_remove):
    flavored_images = []
    for image in images_to_remove:
        if flavor in image:
            flavored_images.append(image.replace(flavor, ''))

    return flavored_images


def get_columns_description(cursor):
    res = dict()
    for ii in range(len(cursor.description)):
        res[cursor.description[ii][0]] = ii
    return res


def generate_dict_according_to_query(query, label, available_images):
    conn = pyodbc.connect(CONNECTION_STRING)
    cursor = conn.cursor()
    cursor.execute(query)
    columns = get_columns_description(cursor)
    patient_id_to_photo_id_dict = {}
    for row in cursor:
        if str(row[columns['id']]) in available_images:
            patient_info = json.loads(row[columns['patientinfo']])
            if row[columns['patientid']] not in patient_id_to_photo_id_dict.keys():
                patient_id_to_photo_id_dict[row[columns['patientid']]] = [
                    (row[columns['id']], label, patient_info['Gendre'])]
            else:
                patient_id_to_photo_id_dict[row[columns['patientid']]].append(
                    (row[columns['id']], label, patient_info['Gendre']))

    return patient_id_to_photo_id_dict


def generate_relevant_image_ids_according_to_query(query, available_images):
    conn = pyodbc.connect(CONNECTION_STRING)
    cursor = conn.cursor()
    cursor.execute(query)
    columns = get_columns_description(cursor)
    images_ids = set()
    for row in cursor:
        if str(row[columns['id']]) in available_images:
            images_ids.add(row[columns['id']])

    return images_ids
