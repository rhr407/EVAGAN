import sys
from PIL import Image


images_list = []

for i in range(1, 151):
    images_list.append('epoch_' + str(i) + '.png')

images = [Image.open(x) for x in images_list]
widths, heights = zip(*(i.size for i in images))

total_width = sum(widths)
max_height = max(heights)

new_im = Image.new('RGB', (total_width, max_height))

x_offset = 0
for im in images:
  new_im.paste(im, (x_offset,0))
  x_offset += im.size[0]

new_im.save('test.png')