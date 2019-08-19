import os
import mmcv


dir_path = os.path.dirname(os.getcwd())

# checkpoints

# pickle_file = dir_path + '/demo/result.pickle'
# pickle_file = dir_path + '/demo/result.bbox.json'
# pickle_file = dir_path + '/datasets/dental_711_2/train.pickle'
# pickle_file = dir_path + '/work_dirs/dental_711_w_pretrained_wt_fix/train_data_result.pickle'
pickle_file = dir_path + '/datasets/dental_711_2/train.pickle'
# pickle_file = dir_path + '/work_dirs/dental_711_w_pretrained_wt_fix/train_data_result.pickle'
# pickle_file = dir_path + '/cleaning/export-2019-07-11T20_41_53_coco_2.pickle'
# pickle_file = dir_path + '/work_dirs/dental_711_w_fix_SSD_classification/test_data_result.pickle'


def main():
    dict = mmcv.load(pickle_file)

    print(len(dict))

    for item in dict:
        if 'cjteidtk9xejb0bqp8zeuzxt4' in item['filename']:
            print(item)

    # for count, item in enumerate(dict):
    #     if item['pigment'] == 1:
    #         print(item)


if __name__ == '__main__':
    main()
