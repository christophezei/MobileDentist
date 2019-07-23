from PIL import Image
import os
import mmcv

dir_path = os.path.dirname(os.getcwd())
image_path = dir_path + '/cleaning/711/'
save_path = dir_path + '/cleaning/711_converted/'

mmcv.mkdir_or_exist(save_path)

for file in os.listdir(image_path):
    print(file)
    img = Image.open(os.path.join(image_path, file))
    img = img.convert('RGB')

    # quality default is 75.
    img.save(os.path.join(save_path, file), quality=75)

