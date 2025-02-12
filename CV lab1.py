#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install cv


# In[2]:


pip list cv


# In[3]:


pip list cv


# In[4]:


get_ipython().system('pip install opencv-python')


# In[5]:


get_ipython().system('pip install opencv-contrib-python')


# In[6]:


import cv2


# In[14]:


image = cv2.imread(r"C:\Users\student\Desktop\images.jpg")
if image is None:
    print("Error!")
else:
        print("image successfully loaded")
cv2.imshow("sample image",image) 
cv2.waitKey(0)
cv2.destroyAllWindows()
gryimage =cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow("Grey scale",gryimage)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(r"C:\Users\student\Desktop\greyscale.jpg",gryimage)
print("image saved successfully")


# In[15]:


image = cv2.imread(r"C:\Users\student\Desktop\hello.jpg")
if image is None:
    print("Error!")
else:
        print("image successfully loaded")


# In[16]:


cv2.imshow("sample image",image) 
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[17]:


gryimage =cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow("Grey scale",gryimage)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[18]:


cv2.imwrite(r"C:\Users\student\Desktop\greyscale12.jpg",gryimage)
print("image saved successfully")


# In[ ]:




