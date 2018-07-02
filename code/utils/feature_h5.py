import glob
import h5py
import argparse
import numpy as np
import os
from tqdm import tqdm


def make_h5(npy_path, h5_path):
  files = glob.glob(npy_path)
  h5_file = h5py.File(h5_path, 'w')
  for _file in tqdm(files):
    vid = os.path.basename(_file)[:-4]
    data = np.load(_file)
    h5_file.create_dataset(vid, data=data)
  h5_file.close()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('npy_path', type=str)
  parser.add_argument('h5_path', type=str)
  args = parser.parse_args()
  
  make_h5(args.npy_path, args.h5_path)
