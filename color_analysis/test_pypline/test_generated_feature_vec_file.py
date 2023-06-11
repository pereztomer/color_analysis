import json


def main():
    file_list = ['/home/tomer/PycharmProjects/ml_testing/json_files/whole_picture/GaussianBlur_radius=9/sRGB/'
                 'sRGB_features_whole_image_GaussianBlur.json',
                 '/home/tomer/PycharmProjects/ml_testing/json_files/'
                 'whole_picture/GaussianBlur_radius=9/Lab/Lab_features_whole_image_GaussianBlur.json']
    for file in file_list:
        try:
            test_file_integrity(file)
        except Exception as e:
            print(e)


def test_file_integrity(file_address):
    f = open(file_address)
    feature_dict = json.load(f)

    list_failed_images = []
    for image_id, feature_vec in feature_dict.items():
        if feature_vec == 'Failed':
            list_failed_images.append(image_id)

    if len(list_failed_images) > 100:
        with open('logs.txt', 'a') as f:
            f.write(f'Failed test,'
                    f'json file: {file_address} failed,'
                    f'number of failed images: {len(list_failed_images)},'
                    f'failed images: {list_failed_images}\n')
    else:
        with open('logs.txt', 'a') as f:
            f.write(f'Passed test,'
                    f'json file: {file_address} failed,'
                    f'number of failed images: {len(list_failed_images)},'
                    f'failed images: {list_failed_images}\n')


if __name__ == '__main__':
    main()
