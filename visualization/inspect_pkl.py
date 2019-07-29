import os
import mmcv


dir_path = os.path.dirname(os.getcwd())

# checkpoints

# pickle_file = dir_path + '/demo/result.pickle'
# pickle_file = dir_path + '/demo/result.bbox.json'
pickle_file = dir_path + '/datasets/dental_711/train.pickle'
# pickle_file = dir_path + '/work_dirs/dental_711_w_pretrained_wt_fix/train_data_result.pickle'
# pickle_file = dir_path + '/datasets/dental_711/train.pickle'
# pickle_file = dir_path + '/work_dirs/dental_711_w_pretrained_wt_fix/train_data_result.pickle'
pickle_file = dir_path + '/cleaning/export-2019-07-11T20_41_53_coco.pickle'


def main():
    dict = mmcv.load(pickle_file)

    print(len(dict))
    print(len(dict[0]))


if __name__ == '__main__':
    main()
