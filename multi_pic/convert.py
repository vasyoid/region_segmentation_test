import numpy as np
from PIL import Image

with open("banana_log.txt") as f:
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
      #suppressedBefore = np.array(info['suppressed_before'], np.bool)
      #suppressedAfter = np.array(info['suppressed_after'], np.bool)
      #outliers = suppressedBefore
      #seemsToBeOutliers = np.logical_and(suppressedAfter, ~suppressedBefore)
      #inliers = ~suppressedAfter
      print(key, len(positions), file = f)
      for val in ids:
          print(val, end = ' ', file = f)
      print('', file = f)
      for val in positions:
          print(int(val[0]), int(val[1]), end = ' ', file = f)
      print('', file = f)
      #for index in range(len(positions)):
      #    type = 0 if inliers[index] else 1 if outliers[index] else 2
      #    print(type, end = ' ', file = f)
      #print('', file = f)
      print((100 * cnt) // len(images_order), '%')
      cnt += 1