import tensorflow as tf
import glob

# updated
def parse_tfr_element(element):
  #use the same structure as above; it's kinda an outline of the structure we now want to create
  data = {
      'raw_image': tf.io.FixedLenFeature([], tf.string),
  }
  content = tf.io.parse_single_example(element, data)
  raw_image = content['raw_image']
  feature = tf.io.parse_tensor(raw_image, out_type=tf.float32)
  return feature


def read_dataset(path, batch_size, shuffle=False, onet_hot=False):
    records_lst = []
    for filename in glob.glob(f'{path}/*', recursive=True):
        records_lst.append(filename)

    dataset = tf.data.TFRecordDataset(filenames=records_lst)
    if shuffle:
        dataset = dataset.map(parse_tfr_element).batch(batch_size=batch_size).shuffle(buffer_size=50)
    else:
        dataset = dataset.map(parse_tfr_element).batch(batch_size=batch_size)
    return dataset


def main():
    train_db_path = '/home/tomer/k_means/Lab_polyp/tf_records'
    batch_size = 1
    train_ds = read_dataset(train_db_path, batch_size)
    for val in train_ds:
        print(val)
        print('hello')


if __name__ == '__main__':
    main()
