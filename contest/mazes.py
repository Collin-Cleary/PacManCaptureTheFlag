import os
for  i in range(20):
  os.system('python maze_generator.py %d > contest%02dCapture.lay' % (i+1, i+1))