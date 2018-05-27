import numpy as np
from PIL import Image

with open("amphora_log.txt") as f:
    lines = f.readlines()
    lines = [line[10+1+8+1:] for line in lines]

data = eval(''.join(lines))
images_order = data['images_order']
images_order.sort()

cnt = 1
with open("output.txt", "w") as f:
  for key in images_order:
      info = data[key]
      positions = np.float32(info['positions'])
      ids = np.int32(info['ids'])
      print(key, len(positions), file = f)
      for val in ids:
          print(val, end = ' ', file = f)
      print('', file = f)
      for val in positions:
          print(int(val[0]), int(val[1]), end = ' ', file = f)
      print('', file = f)
      print((100 * cnt) // len(images_order), '%')
      cnt += 1