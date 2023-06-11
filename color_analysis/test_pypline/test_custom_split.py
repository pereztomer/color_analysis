import json

import pandas as pd
import pyodbc

CONNECTION_STRING = 'Driver={ODBC Driver 18 for SQL Server};Server=tcp:gixamsqlserver.database.windows.net,1433;Database=gixammain;Persist Security Info=False;UID=admindb;PWD=ItsG00dT0BeH0me;MultipleActiveResultSets=False;Connection Timeout=80;'


def get_columns_description(cursor):
    res = dict()
    for ii in range(len(cursor.description)):
        res[cursor.description[ii][0]] = ii
    return res


def main():
    json_files_path = '/home/tomer/GitProjects/DeepLearning/Data/color_analysis_json_files/kmeans_centroids_24/' \
                      'percentage_feature_vec/preprocess_C/full_image/Lab/gaussian_blur_False/' \
                      'approved=1_or_approved=2_or_approved=0/weird_train_validation_color_feature_vec.json'

    f = open(json_files_path)
    data_set = json.load(f)
    del data_set['params']

    keys = list(data_set.keys())
    relevant_keys = {'approved': 3, 'deleted': 9, 'position': 13, 'C_CRC': 20, 'C_PolypATLG1': 21, 'C_PolypATGT1': 22,
                     'C_PolypSLT1': 23, 'C_PolypSGT1': 24, 'C_PolypHyperPlasty': 25, 'C_Colitis': 26,
                     'G_GastritisHP': 27,
                     'G_EsophogusA': 28, 'G_EsophogusBCD': 29, 'G_Baret': 30, 'G_Metaplasia': 31, 'G_PolypATGT1': 32,
                     'G_PolypATLT1': 33, 'G_PolypHLT1': 34, 'G_PolypHGT1': 35, 'G_Celiac': 36, 'G_GIST': 37,
                     'G_StomachCancer': 38, 'G_GastricLymph': 39, 'locked': 40, 'colonoscopy': 43, 'gastroscopy': 44,
                     'rejected': 45, 'gender': 61}

    polyp_keys = [key for key, value in data_set.items() if value[1] == 1]
    # polyp_patient_ids = [x.split(',')[0].replace('(', '') for x in polyp_keys]
    polyp_file_name = 'statistics_polyp'
    statistics_polyp = generate_vals_list(polyp_keys, relevant_keys, polyp_file_name)
    with open(polyp_file_name, "w") as outfile:
        outfile.write(json.dumps(statistics_polyp, indent=4))

    healthy_keys = [key for key, value in data_set.items() if value[1] == 0]
    # healthy_patient_ids = [x.split(',')[0].replace('(', '') for x in healthy_keys]
    healthy_file_name = 'statistics_healthy'
    statistics_healthy = generate_vals_list(healthy_keys, relevant_keys, healthy_file_name)
    with open('statistics_healthy', "w") as outfile:
        outfile.write(json.dumps(statistics_healthy, indent=4))

    #

    # # str((int(patient_id), photo_id))
    #
    # healthy_statistics = calc_statistics(healthy_patient_ids)
    # healthy_statistics.to_csv('healthy_statistics.csv')
    # print('Healthy: ')
    # print(healthy_statistics)
    #
    # print('=========================================================')
    # polyp_statistics = calc_statistics(polyp_patient_ids)
    # print('Polyp: ')
    # print(polyp_statistics)
    #
    # polyp_statistics.to_csv('polyp_statistics.csv')


# def calc_statistics(patient_ids):
#     query = f'select * from gixammain.dbo.ImageStudyPatient where patientid in {patient_ids}'.replace("[", "(").replace(
#         "]", ")")
#     conn = pyodbc.connect(CONNECTION_STRING)
#     cursor = conn.cursor()
#     cursor.execute(query)
#     columns = get_columns_description(cursor)
#     df_init_values = {col_name: [0] for col_name in columns.keys()}
#     df_statistics = pd.DataFrame(df_init_values)
#
#     for row in cursor:
#         df_statistics = update_df(df_statistics, row, columns)
#
#     return df_statistics


def update_df(statictics_dict, row, columns):
    for key in statictics_dict:
        if row[columns[key]] is True:
            statictics_dict[key] += 1
        if key == 'approved':
            statictics_dict['approved'][row[columns[key]]] += 1
    return statictics_dict


def generate_vals_list(keys_list, relevant_keys, file_name):
    got_desc = False
    statistics_dict = {col_name: 0 for col_name in relevant_keys}
    statistics_dict['approved'] = {1: 0, 2: 0, 0: 0}
    print(f'total number of key in {file_name} is: {len(keys_list)}')
    for idx, key in enumerate(keys_list):
        patinetid = key.split(',')[0].replace('(', '')
        id = key.split(',')[1].replace(')', '')
        query = f'select * from gixammain.dbo.ImageStudyPatient where id = {id} and patientid = {patinetid}'
        conn = pyodbc.connect(CONNECTION_STRING)
        cursor = conn.cursor()
        cursor.execute(query)
        if got_desc is False:
            columns = get_columns_description(cursor)
            got_desc = True

        for row in cursor:
            statistics_dict = update_df(statistics_dict, row, columns)

        if idx % 100 == 0:
            print(f'passed {idx} values')
            with open(file_name, "w") as outfile:
                outfile.write(json.dumps(statistics_dict, indent=4))
    return statistics_dict


if __name__ == '__main__':
    main()
