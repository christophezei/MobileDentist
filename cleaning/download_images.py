import os
import mmcv
import urllib.request

dir_path = os.path.dirname(os.getcwd())

labelbox_file = dir_path + '/cleaning/export-2019-07-11T20_41_53.065Z.json'
out_dir = dir_path + '/cleaning/711_with_rotate/'


def main():
    mmcv.mkdir_or_exist(out_dir)

    items = mmcv.load(labelbox_file)
    print('downloading {} images. '.format(len(items)))

    count = 1
    for item in items:
        name = item['DataRow ID']
        image_link = item['Labeled Data']

        # img_data = requests.get(image_link).content
        # with open(out_dir+'{}.jpg'.format(name), 'wb') as handler:
        #     handler.write(img_data)
        urllib.request.urlretrieve(image_link, out_dir+'{}'.format(name))

        print(count)
        count = count + 1


if __name__ == '__main__':
    main()
