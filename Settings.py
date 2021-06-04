import numpy as np

#settings for line detection
VERTICES = np.array([[300, 700], [320, 360] , [450, 360],[500, 700]])
SRC= np.float32([(0.35, 0.52), (0.62, 0.52), (0.02, 0.75), (0.98, 0.75)])
DST=np.float32([(0,0), (1, 0), (0,1), (1,1)])
MONITOR = {"top": 20, "left": 80, "width": 800, "height": 620}
NWINDOWS = 25
MARGINES=50
MIN_PIX = 10
frameWidth= 800
frameHeight = 600

#settings for Keys getter
KEYS = {'up':'w', 'left':'a', 'right':'d', 'down':'s'}
DIRECTION_CLASS = {'up':0, 'left':1, 'right':2, 'left-up':3, 'right-up':4, 'down':5, 'nothing': 6}






