from mmcv.image import imread, imwrite
import os

dir_path = os.path.dirname(os.getcwd())
# file_path = dir_path + '/cleaning/711_with_rotate_2/'
file_path = dir_path + '/cleaning/711/'

# one row image
# image_path = file_path + 'cjtirr3c2amnf0bqpx4khl7ea.jpg'
# img = imread(image_path)
# height, width, _ = img.shape
# print('row image')
# print('height: {}'.format(height))
# print('width: {}'.format(width))
#
# # one column image
# image_path = file_path + 'cjtjsecwubr8n0bqpzzk4i4t7.jpg'
# img = imread(image_path)
# height, width, _ = img.shape
# print('column image')
# print('height: {}'.format(height))
# print('width: {}'.format(width))


from PIL import Image

# one row image
image_path = file_path + 'cjtirr3c2amnf0bqpx4khl7ea.jpg'
img = Image.open(image_path)
width, height = img.size
print('row image')
print('height: {}'.format(height))
print('width: {}'.format(width))

# one column image
image_path = file_path + 'cjtjsecwubr8n0bqpzzk4i4t7.jpg'
img = Image.open(image_path)
width, height = img.size
print('column image')
print('height: {}'.format(height))
print('width: {}'.format(width))