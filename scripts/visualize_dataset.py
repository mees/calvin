import cv2
import numpy as np

from argparse import ArgumentParser
from pathlib  import Path


if __name__ == '__main__':
    parser = ArgumentParser(description='Interactive visualization of CALVIN dataset')
    parser.add_argument('path', type=str, help='Path to dir containing scene_info.npy')
    parser.add_argument('-d', '--data', nargs='*', default=['rgb_static', 'rgb_gripper'], help='Data to visualize')
    args = parser.parse_args()

    if not Path(args.path).is_dir():
        print(f'Path {args.path} is either not a directory, or does not exist.')
        exit()

    files = sorted(Path(args.path).glob('episode_*.npz'))

    idx = 0

    while True:
        t = np.load(files[idx], allow_pickle=True)

        for d in args.data:
            if d not in t:
                print(f'Data {d} cannot be found in transition')
                continue
            
            cv2.imshow(d, t[d])
        
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
        elif key == 83: #  Right arrow or p
            idx = (idx + 1) % len(files)
        elif key == 81: #  Left arrow or o
            idx = (len(files) + idx - 1) % len(files)
        else:
            print(f'Unrecognized keycode "{key}"')
