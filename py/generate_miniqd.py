import argparse
import numpy as np
import glob
import os
import pdb
from tqdm import tqdm

# loops over all numpy files in quickdraw
# takes a slice of [:num_examples]
# saves the new slice in a new folder with the same name

TOTAL_CLASSES = 345
def generate_mini_quickdraw(args):
    print(args)

    all_file_paths = glob.glob(
            os.path.join(args.base_dir, '*'))
    assert len(all_file_paths) == TOTAL_CLASSES
    
    for fp in tqdm(all_file_paths):
        new_path = os.path.join(args.target_dir, fp[len(args.base_dir):])
        if os.path.isfile(new_path) and not args.overwrite:
            if args.verbose:
                print(f"{new_path} exists, skipping")
            continue
        if args.verbose:
            print(f"Generating {new_path}")
        x_sub = generate_subsample(fp, args)
        np.save(new_path, x_sub)
        del x_sub

def generate_subsample(np_path, args):
    x = np.load(np_path)
    x_sub = x[np.random.choice(range(len(x)), size=args.num_samples)]
    return x_sub

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Generate mini_quickdraw')
    parser.add_argument('--num_samples', '-ns' , type=int, default=100,
                        help='Num subsamples for each class')
    parser.add_argument('--overwrite', '-ow', default=False, action='store_true',
                        help='Whether to overwrite existing files')
    parser.add_argument('--verbose', '-v', default=False, action='store_true',
                        help='Whether to print status')
    parser.add_argument('--base_dir', type=str, default='./data/quickdraw/',
                        help='directory to load from')
    parser.add_argument('--target_dir', type=str, default='./data/mini_quickdraw/',
                        help='directory to save to')
    

    args = parser.parse_args()
    generate_mini_quickdraw(args)