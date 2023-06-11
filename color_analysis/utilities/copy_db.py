import glob
from tqdm import tqdm
import shutil


def main():
    total_addresses = glob.glob(f'/media/beast/WD12T/PreProcessed_1_7_2022/**.png')
    with tqdm(total=len(total_addresses)) as pbar:
        for address in total_addresses:
            image_index = address.split('/')[-1].replace('.png', '')
            flavor = image_index[0]
            if flavor == 'B':
                continue
            shutil.copy(address, '/home/beast/GitProjects/DeepLearning/full_db_flavor_C')
            pbar.update(1)


if __name__ == '__main__':
    main()
