# -*- coding: utf-8 -*-
"""
Advanced_Machine_Learning Project

@author: Ruihao Wang
"""

import numpy as np
import matplotlib.pyplot as plt

cam1 = np.load('heatmap_for_dog.npy')
cam2 = np.load('heatmap_for_cat.npy')
cam3 = np.load('heatmap_for_horse.npy')
cam4 = np.load('heatmap_for_bird.npy')
plt.imshow(cam1)
plt.colorbar()
plt.show()
plt.imshow(cam2)
plt.colorbar()
plt.show()
plt.imshow(cam3)
plt.colorbar()
plt.show()
plt.imshow(cam4)
plt.colorbar()
plt.show()
