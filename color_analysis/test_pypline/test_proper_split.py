import json
import pyodbc

CONNECTION_STRING = 'Driver={ODBC Driver 18 for SQL Server};Server=tcp:gixamsqlserver.database.windows.net,1433;' \
                    'Database=gixammain;Persist Security Info=False;UID=admindb;PWD=ItsG00dT0BeH0me;' \
                    'MultipleActiveResultSets=False;Connection Timeout=80;'


def get_columns_description(cursor) -> dict:
    """
    utility function for easy access to cursor row via keys and not indexes
    :param cursor: pyodbc object
    :return: dictionary containing key->index mapping
    """
    res = dict()
    for ii in range(len(cursor.description)):
        res[cursor.description[ii][0]] = ii
    return res


def test_json_file(folder_path, json_file, query_healthy, query_polyp, approved_query, balanced_class) -> None:
    """
    utility function for test proper split
    :param folder_path: folder of the json files created
    :param json_file: json file name
    :param query_healthy: a query defining a healthy patient
    :param query_polyp: a query defining a polyp patient
    :param approved_query: values of approved in the sql query
    :param balanced_class: a flag indicating if classes are supposed to be balanced
    :return: None
    """
    f = open(f'{folder_path}/{approved_query.replace(" ", "_")}/train_validation_{json_file}')
    train_validation_feature_vectors_dict = json.load(f)
    del train_validation_feature_vectors_dict['params']
    for key, value in train_validation_feature_vectors_dict.items():
        if value[0] == 'Failed':
            raise Exception('Failed vector in json file')

    train_validation_patient_ids = [x.split(',')[0].replace('(', '') for x in
                                    train_validation_feature_vectors_dict.keys()]

    train_validation_image_ids = [x.split(',')[1].replace(')', '') for x in
                                  train_validation_feature_vectors_dict.keys()]

    assert len(train_validation_image_ids) == len(set(train_validation_patient_ids))
    # 1 indicates the patient has some sort of polyp
    # 0 indicates the patient has no polyps

    conn = pyodbc.connect(CONNECTION_STRING)
    cursor = conn.cursor()
    cursor.execute(query_healthy)
    columns = get_columns_description(cursor)
    healthy_counter_train_validation = 0
    polyp_counter_train_validation = 0

    for row in cursor:
        key = str((int(row[columns['patientid']]), row[columns['id']]))
        if key in train_validation_feature_vectors_dict:
            if train_validation_feature_vectors_dict[key][1] == 1:
                raise Exception('Wrong label (label=1) for a healthy patient in train_validation')
            else:
                healthy_counter_train_validation += 1

    conn = pyodbc.connect(CONNECTION_STRING)
    cursor = conn.cursor()
    cursor.execute(query_polyp)
    columns = get_columns_description(cursor)
    for row in cursor:
        key = str((int(row[columns['patientid']]), row[columns['id']]))
        if key in train_validation_feature_vectors_dict:
            if train_validation_feature_vectors_dict[key][1] == 0:
                raise Exception('Wrong label (label=0) for a polyp patient in train_validation')
            else:
                polyp_counter_train_validation += 1
    if balanced_class:
        assert healthy_counter_train_validation == polyp_counter_train_validation

    print(f'Test for {folder_path}/{json_file} succeeded')

def main():
    test_json_file()


if __name__ == '__main__':
    main()
