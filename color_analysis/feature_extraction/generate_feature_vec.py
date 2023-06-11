import glob
import json
from tqdm import tqdm
import os
import lab_feature_extraction_numba_cuda
import rgb_feature_extraction_numba_cuda
import datetime
import git
import numpy as np


def extract_feature_vectors(source_path: str, dest_path: str, json_file_name: str, gaussian_blur: bool, image_cut: str,
                            color_space: str, flavor: str, specific_color_centroids: np.array):
    """
     extract feature vectors for all images in the source path
     :param source_path: images path
     :param dest_path: path of the json file to written at
     :param json_file_name: name of the json file
     :param gaussian_blur: a boolean value indicating if to pass a gaussian blur on the images
     :param image_cut: what cut to cut the given image
     :param color_space: sRGB/Lab
     :param flavor: preprocessing type
     :param specific_color_centroids: centroids to calculates feature vector from
     :return: None
     """
    assert color_space == 'sRGB' or color_space == 'Lab'
    os.makedirs(dest_path, exist_ok=True)
    if os.path.exists(f'{dest_path}/{json_file_name}'):
        f = open(f'{dest_path}/{json_file_name}')
        feature_vectors_dict = json.load(f)
        assert 'params' in feature_vectors_dict.keys()
        assert feature_vectors_dict['params']['color_space'] == color_space
        assert feature_vectors_dict['params']['gaussian_blur'] == gaussian_blur
        assert feature_vectors_dict['params']['image_cut'] == image_cut
        assert feature_vectors_dict['params']['image_flavor'] == flavor
        assert 'centroids' in feature_vectors_dict['params']
    else:
        feature_vectors_dict = {}
        now = datetime.datetime.now()
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        params_dict = {"color_space": color_space, "gaussian_blur": gaussian_blur, "image_cut": image_cut,
                       "image_flavor": flavor, "time": now.strftime("%Y-%m-%d %H:%M:%S"),
                       "feature_extraction_git_version": sha, 'centroids': specific_color_centroids.tolist()}
        feature_vectors_dict['params'] = params_dict

    total_addresses = glob.glob(f'{source_path}/**.png')
    with tqdm(total=len(total_addresses)) as pbar:
        for idx, address in enumerate(total_addresses):
            image_flavor = address.split('/')[-1].replace('.png', '')[
                0]  # flavor assumed to be the first letter in the image file name
            if flavor == image_flavor:
                image_id = address.split('/')[-1].replace('.png', '').replace(flavor, '')
                if image_id in feature_vectors_dict:
                    pbar.update(1)
                    continue
                try:
                    if color_space == 'sRGB':
                        feature_vec = rgb_feature_extraction_numba_cuda.calc_image_feature_vec_cuda_wrapper(
                            address=address,
                            gaussian_blur=gaussian_blur,
                            image_cut=image_cut)
                    elif color_space == 'Lab':
                        feature_vec = lab_feature_extraction_numba_cuda.calc_image_feature_vec_cuda_wrapper(
                            address=address,
                            gaussian_blur=gaussian_blur,
                            image_cut=image_cut,
                            centroids=specific_color_centroids)
                    feature_vectors_dict[image_id] = feature_vec.tolist()
                except Exception as e:
                    feature_vectors_dict[image_id] = 'Failed'
                    print(e)
                if idx % 10 == 0:
                    with open(f'{dest_path}/{json_file_name}', "w") as outfile:
                        outfile.write(json.dumps(feature_vectors_dict, indent=4))
                pbar.update(1)



def main():
    # healthy_centroids_address = '/home/tomer/GitProjects/DeepLearning/Data/custom_kmeans/Lab_full_image_noBlur/healthy/' \
    #                             '250_pics_stats_points_for_kmeans_k_25.json'
    #
    # polyp_centroids_address = '/home/tomer/GitProjects/DeepLearning/Data/custom_kmeans/Lab_full_image_noBlur/polyp/' \
    #                           '250_pics_stats_points_for_kmeans_k_25.json'
    #
    # # load centroids
    # f1 = open(healthy_centroids_address)
    # lab_healthy_centroids = np.array(json.load(f1), dtype=np.float32)
    #
    # f2 = open(polyp_centroids_address)
    # lab_polyp_centroids = np.array(json.load(f2), dtype=np.float32)
    # lab_color_centroids = np.concatenate([lab_healthy_centroids, lab_polyp_centroids])


    centroids_address = '/home/beast/GitProjects/DeepLearning/data/k_means/Lab_full_image_noBlur/healthy_and_polyp/' \
                        '200_pics_stats_points_for_kmeans_k_25.json'
    lab_color_centroids = np.array(json.load(open(centroids_address)))
    color_centroids = {('gaussian_blur_False', 'full_image', 'Lab', 'flavor_C'): lab_color_centroids}

    source_path = '/home/beast/GitProjects/DeepLearning/full_db_flavor_C'
    destination_folder = '/home/beast/GitProjects/DeepLearning/data'
    destination_folder = f'{destination_folder}/color_analysis_json_files/kmeans_centroids_{lab_color_centroids.shape[0]}/percentage_feature_vec'
    print(destination_folder)
    exit()
    os.makedirs(destination_folder, exist_ok=True)
    flavor_list = ['C']
    gaussian_blur_lst = [False]
    image_cut_lst = ['full_image']
    color_space_lst = ['Lab']  # ['sRGB', 'Lab']

    for gaussian_blur in gaussian_blur_lst:
        for image_cut in image_cut_lst:
            for color_space in color_space_lst:
                for flavor in flavor_list:
                    specific_color_centroid_key = (
                        f'gaussian_blur_{gaussian_blur}', image_cut, color_space, f'flavor_{flavor}')
                    assert specific_color_centroid_key in color_centroids
                    specific_color_centroid = color_centroids[specific_color_centroid_key]
                    full_destination_path = f'{destination_folder}/preprocess_{flavor}/{image_cut}/{color_space}/gaussian_blur_{gaussian_blur}'
                    json_file_name = 'color_feature_vec.json'
                    extract_feature_vectors(source_path, full_destination_path, json_file_name, gaussian_blur,
                                            image_cut, color_space, flavor, specific_color_centroid)


if __name__ == '__main__':
    main()
