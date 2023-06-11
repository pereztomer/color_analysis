from sklearn.model_selection import StratifiedKFold
import glob
import json
import random
import os
from Code.color_analysis.utilities.custom_utils import remove_images, chose_flavor, generate_dict_according_to_query
from Code.color_analysis.test_pypline.test_proper_split import test_json_file
import numpy as np


def generate_relevant_images(file_address: str, quality_measures_params: dict):
    """
    reading a json file containing for each of the photos from the dataset a feature vector.
    removing unwanted images according to different metrics
    :param file_address: string address to the feature vector file
    :param quality_measures_params: a dictionary containing different parameters regarding quality measures
    :return: a dictionary containing relevant image and a dictionary containing parameters of the json file
    """
    f = open(file_address)
    feature_vectors_dict = json.load(f)

    params_dict = feature_vectors_dict['params']
    del feature_vectors_dict['params']
    sharpness_threshold = quality_measures_params['sharpness_threshold']
    pixels_std_deviations = quality_measures_params['pixels_std_deviations']
    num_of_reflection_std_deviations = quality_measures_params['num_of_reflection_std_deviations']
    flavor = 'C'  # params_dict['flavor']
    quality_measures_address = quality_measures_params['quality_measures_address']

    images_to_remove = remove_images(quality_measures_address=quality_measures_address,
                                     sharpness_threshold=sharpness_threshold,
                                     num_of_pixel_std_deviations=pixels_std_deviations,
                                     num_of_reflection_std_deviations=num_of_reflection_std_deviations)

    images_to_remove = chose_flavor(flavor, images_to_remove)

    for image_id, feature_vec in feature_vectors_dict.items():
        if feature_vec == 'Failed':
            images_to_remove.append(image_id)

    images_to_remove = list(set(images_to_remove))
    clean_feature_vectors_dict = {}
    for key, value in feature_vectors_dict.items():
        if key in images_to_remove:
            continue
        clean_feature_vectors_dict[key] = value

    return clean_feature_vectors_dict, params_dict


def remove_conflicting_patient_ids(healthy_patient_id_to_photo_id_dict: dict, polyp_patient_id_to_photo_id_dict: dict):
    """
    remove ids that are both in the polyp group and in the healthy group
    :param healthy_patient_id_to_photo_id_dict: a string address to folder of the feature vector file
    :param polyp_patient_id_to_photo_id_dict: feature vector file name
    :return: 2 dictionaries healthy and polyp
    """

    ids_to_remove = set()
    for polyp_id in polyp_patient_id_to_photo_id_dict:
        if polyp_id in healthy_patient_id_to_photo_id_dict:
            ids_to_remove.add(polyp_id)

    for value in ids_to_remove:
        del healthy_patient_id_to_photo_id_dict[value]
        del polyp_patient_id_to_photo_id_dict[value]

    for polyp_id in polyp_patient_id_to_photo_id_dict:
        if polyp_id in healthy_patient_id_to_photo_id_dict:
            raise Exception('Same id in polyp class and healthy class')

    return healthy_patient_id_to_photo_id_dict, polyp_patient_id_to_photo_id_dict


def split_data_set(folder_address: str, json_file_name: str, query_healthy: str, query_polyp: str,
                   quality_measures_params: dict, approved_query: str) -> None:
    """
    generating train_validation splits
    :param folder_address: a string address to folder of the feature vector file
    :param json_file_name: feature vector file name
    :param query_healthy: a sql query indicating what a healthy patient is
    :param query_polyp: a sql query defining patients which polyp
    :param quality_measures_params: a dictionary containing different parameters regarding quality measures
    :param approved_query: values of approved in the sql query
    :return: None
    """
    approved_query = approved_query.replace(" ", "_")
    clean_feature_vectors_dict, params_dict = generate_relevant_images(f'{folder_address}/{json_file_name}',
                                                                       quality_measures_params)
    params_dict['query_healthy'] = query_healthy
    params_dict['query_polyp'] = query_polyp
    params_dict['approved_vals'] = approved_query

    # 1 indicates the patient has some sort of polyp
    # 0 indicates the patient has no polyps

    healthy_patient_id_to_photo_id_dict = generate_dict_according_to_query(query=query_healthy,
                                                                           available_images=clean_feature_vectors_dict,
                                                                           label=0)
    polyp_patient_id_to_photo_id_dict = generate_dict_according_to_query(query=query_polyp,
                                                                         available_images=clean_feature_vectors_dict,
                                                                         label=1)

    healthy_patient_id_to_photo_id_dict, polyp_patient_id_to_photo_id_dict = remove_conflicting_patient_ids(
        healthy_patient_id_to_photo_id_dict, polyp_patient_id_to_photo_id_dict)

    # choose randomly the image id for a given patient
    for key, value in polyp_patient_id_to_photo_id_dict.items():
        chosen_value = random.choice(value)
        polyp_patient_id_to_photo_id_dict[key] = chosen_value

    for key, value in healthy_patient_id_to_photo_id_dict.items():
        chosen_value = random.choice(value)
        healthy_patient_id_to_photo_id_dict[key] = chosen_value

    patient_id_to_photo_id_dict = {}
    patient_id_to_photo_id_dict.update(polyp_patient_id_to_photo_id_dict)
    patient_id_to_photo_id_dict.update(healthy_patient_id_to_photo_id_dict)

    train_validation_dict = {'params': params_dict}
    for patient_id in patient_id_to_photo_id_dict:
        photo_id, label = patient_id_to_photo_id_dict[patient_id]
        feature_vec = clean_feature_vectors_dict[str(photo_id)]
        train_validation_dict[str((int(patient_id), photo_id))] = [feature_vec, label]

    os.makedirs('/home/tomer/test_kfold', exist_ok=True)
    with open(f'/home/tomer/test_kfold/train_validation_{json_file_name}', "w") as outfile:
        outfile.write(json.dumps(train_validation_dict, indent=4))


def main():
    random.seed(100)
    approved_values_list = [["1"], ["1", "2"], ["1", "2", "0"]]

    ds_path = '/home/tomer/GitProjects/DeepLearning/Data/color_analysis_json_files/full_image/sRGB/gaussian_blur_True'
    ds_json_files = glob.glob(f'{ds_path}/**/color_feature_vec.json', recursive=True)

    quality_measures_params = {'sharpness_threshold': 370000,
                               'pixels_std_deviations': 2.5,
                               'num_of_reflection_std_deviations': 2.5,
                               'quality_measures_address': '/home/tomer/Jubban/quality_measures'}
    for approved_val in approved_values_list:
        approved_query = [f'approved={val}' for val in approved_val]
        approved_query = f' or '.join(approved_query)
        query_healthy = f'select id, patientid from gixammain.dbo.ImageStudyPatient where deleted = 0 ' \
                        f'and rejected = 0 and ({approved_query}) and colonoscopy=1 and C_CRC=0 and C_PolypATLG1=0 and ' \
                        f'C_PolypATGT1=0 and C_PolypSLT1=0 and C_PolypSGT1=0 and C_PolypHyperPlasty=0 ' \
                        f'and C_Colitis=0 and [position] = 0 and locked = 1 and branchid in (4,6,8)'

        query_polyp = f'select id, patientid from gixammain.dbo.ImageStudyPatient where deleted = 0 and rejected = 0 ' \
                      f'and C_PolypATLG1 = 1 and ({approved_query}) and [position] = 0 and locked = 1 and branchid in (4,6,8)'

        for j_file in ds_json_files:
            folder_address, json_file_name = os.path.split(os.path.abspath(j_file))
            try:
                split_data_set(folder_address, json_file_name, query_healthy, query_polyp,
                               quality_measures_params, approved_query)
            except Exception as e:
                print('Failed generating train_validation json file')
                print(e)
            try:
                test_json_file(folder_address, json_file_name, query_healthy, query_polyp, approved_query)
            except Exception as e:
                print('Failed json file test')
                print(e)
            exit()


def prepare_ds(ds_address):
    f = open(ds_address)
    train_val_dict = json.load(f)
    json_file_params = train_val_dict['params']
    del train_val_dict['params']
    train_validation_x = []
    train_validation_y = []
    for key, value in train_val_dict.items():
        train_validation_x.append(value[0])
        train_validation_y.append(value[1])
    train_validation_x = np.array(train_validation_x)
    train_validation_y = np.array(train_validation_y)

    return train_validation_x, train_validation_y, json_file_params


def create_k_fold(x, y, fold_num, seed):
    # 1 indicates the patient has some sort of polyp
    # 0 indicates the patient has no polyps

    random.seed(seed)
    total_ds = np.array(list(zip(x, y)))

    healthy_indices_list = [idx for idx, sample in enumerate(total_ds) if sample[1] == 0]
    polyp_indices_list = [idx for idx, sample in enumerate(total_ds) if sample[1] == 1]
    assert len(healthy_indices_list) > len(polyp_indices_list)
    random.shuffle(polyp_indices_list)
    polyp_fold_num_split = np.array_split(polyp_indices_list, fold_num)

    random.shuffle(healthy_indices_list)
    healthy_validation_values = healthy_indices_list[:len(polyp_indices_list)]
    healthy_remainder_values = healthy_indices_list[len(polyp_indices_list):]
    healthy_fold_num_split = np.array_split(healthy_validation_values, fold_num)

    for split_num in range(fold_num):
        split_validation = []
        split_validation.extend(polyp_fold_num_split[split_num].tolist())
        split_validation.extend(healthy_fold_num_split[split_num].tolist())

        split_train_healthy = [sample.tolist() for idx, sample in enumerate(healthy_fold_num_split) if idx != split_num]
        flat_split_train_healthy = [item for sublist in split_train_healthy for item in sublist]
        flat_split_train_healthy.extend(healthy_remainder_values)

        split_train_polyp = [sample.tolist() for idx, sample in enumerate(polyp_fold_num_split) if idx != split_num]
        flat_split_train_polyp = [item for sublist in split_train_polyp for item in sublist]
        enlarge_size = len(flat_split_train_healthy) // len(flat_split_train_polyp) + 1

        split_train = [ele for ele in flat_split_train_polyp for _ in range(enlarge_size)]
        split_train.extend(flat_split_train_healthy)
        yield split_train, split_validation


if __name__ == '__main__':
    # main()
    file_address = '/home/tomer/test_kfold/train_validation_color_feature_vec.json'
    train_validation_x, train_validation_y, json_file_params = prepare_ds(file_address)

    values = create_k_fold(x=train_validation_x, y=train_validation_y, fold_num=5, seed=150)
    for fold in values:
        print(fold)
