import os
import mmcv


dir_path = os.path.dirname(os.getcwd())

# checkpoints

# pickle_file = dir_path + '/demo/result.pickle'
# pickle_file = dir_path + '/demo/result.bbox.json'
pickle_file = dir_path + '/datasets/dental_711/test.pickle'


def main():
    dict = mmcv.load(pickle_file)

    print(dict)


if __name__ == '__main__':
    main()
