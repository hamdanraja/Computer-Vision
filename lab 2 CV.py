#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install opencv-python')
get_ipython().system('pip install opencv-python-headless')


# Task 1: Load an image, convert it to grayscale, and display it.

# In[2]:


import cv2
import numpy as np
from matplotlib import pyplot as plt
img= cv2.imread(r"C:\Users\student\Desktop\lion.jpg")
img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.axis('off')
plt.show()


# In[4]:


gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(gray_img,cmap='gray')
plt.axis('off')
plt.show()


# In[6]:


resize_img=cv2.resize(img,(400,400))
plt.imshow(cv2.cvtColor(resize_img,cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()


# In[8]:


import cv2
import numpy as np
img= cv2.imread(r"C:\Users\student\Desktop\lion.jpg")
plt.imshow(img,cmap='gray')
plt.title("origional grey scale image")
plt.axis('off')
plt.show()


# Task 2: Create the negative of the grayscale image and display the result.

# In[9]:


negative_img=255-img
plt.imshow(img,cmap='gray')
plt.title("origional grey scale image")
plt.axis('off')
plt.show()


# #Task 3: Convert the grayscale image into a binary image using thresholding and display the binary image.

# In[11]:


_,binary_img=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
plt.imshow(binary_img,cmap='gray')
plt.title("binary image")
plt.axis('off')
plt.show()


# Task 4: Create and display the negative of the binary image.

# In[12]:


_,negative_binary_img=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
plt.imshow(negative_binary_img,cmap='gray')
plt.title("negative_binary image")
plt.axis('off')
plt.show()


# Task Extension: Experiment with different threshold values in the binary thresholding step.

# In[14]:


for threshold_value in [100, 150, 200]:
    _, binary_image = cv2.threshold(gray_img, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Display the binary image for each threshold value
    plt.imshow(binary_image, cmap='gray')
    plt.title(f"Binary Image with Threshold {threshold_value}")
    plt.axis('off')  # Hide axis
    plt.show()


# In[ ]:




