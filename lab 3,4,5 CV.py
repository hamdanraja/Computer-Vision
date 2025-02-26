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


# In[3]:


gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(gray_img,cmap='gray')
plt.axis('off')
plt.show()


# In[4]:


resize_img=cv2.resize(img,(400,400))
plt.imshow(cv2.cvtColor(resize_img,cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()


# In[5]:


import cv2
import numpy as np
img= cv2.imread(r"C:\Users\student\Desktop\lion.jpg")
plt.imshow(img,cmap='gray')
plt.title("origional grey scale image")
plt.axis('off')
plt.show()


# Task 2: Create the negative of the grayscale image and display the result.

# In[6]:


negative_img=255-img
plt.imshow(img,cmap='gray')
plt.title("origional grey scale image")
plt.axis('off')
plt.show()


# #Task 3: Convert the grayscale image into a binary image using thresholding and display the binary image.

# In[7]:


_,binary_img=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
plt.imshow(binary_img,cmap='gray')
plt.title("binary image")
plt.axis('off')
plt.show()


# Task 4: Create and display the negative of the binary image.

# In[8]:


_,negative_binary_img=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
plt.imshow(negative_binary_img,cmap='gray')
plt.title("negative_binary image")
plt.axis('off')
plt.show()


# Task Extension: Experiment with different threshold values in the binary thresholding step.

# In[9]:


for threshold_value in [100, 150, 200]:
    _, binary_image = cv2.threshold(gray_img, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Display the binary image for each threshold value
    plt.imshow(binary_image, cmap='gray')
    plt.title(f"Binary Image with Threshold {threshold_value}")
    plt.axis('off')  # Hide axis
    plt.show()


# In[12]:


from PIL import Image
import cv2
img= cv2.imread(r"C:\Users\student\Desktop\lion.jpg")
gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(gray_img,cmap='gray')
plt.title('orgional gray scale image')
plt.axis('off')
plt.show()


# In[13]:


hist=cv2.calcHist([gray_img],[0],None,[256],[0,256])
plt.plot(hist)
plt.title('histogram')
plt.xlabel('pixel intensity')
plt.ylabel('frequency')


# In[16]:


import cv2
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread(r"C:\Users\student\Desktop\lion.jpg")

# Convert to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Ensure the image is of type uint8 (8-bit)
gray_img = gray_img.astype('uint8')

# Apply histogram equalization
equalized_image = cv2.equalizeHist(gray_img)

# Display the result
plt.imshow(equalized_image, cmap='gray')
plt.title('Equalized Image')
plt.axis('off')
plt.show()


# In[17]:


equalized_hist=cv2.calcHist([equalized_image],[0],None,[256],[0,256])
plt.plot(equalized_hist)
plt.title('histogram')
plt.xlabel("pixel")
plt.ylabel('frequency')
plt.show()


# In[22]:


def show_bit_planes(img):
    bit_planes=[]
    for i in range(8):
        bit_plane=(img & (1<<i))>>i
        bit_planes.append(bit_plane*255)
    fig,axes=plt.subplots(2,4,figsize=(12,6))
    axes=axes.ravel()
    for i in range(8):
        axes[i].imshow(bit_planes[i], cmap='gray')
        axes[i].set_title(f'Bit Plane{i}')
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()
show_bit_planes(gray_img)            


# ## LAB 5

# In[25]:


import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_transformation(image, matrix):
    rows, cols = image.shape[:2]  # Indent this line
    transformed_image = cv2.warpAffine(image, matrix, (cols, rows))  # Indent this line
    return transformed_image  # Indent this line

# Read the image in grayscale
image = cv2.imread(r"C:\Users\student\Desktop\lion.jpg", 0)

# Translation
Tx, Ty = 50, 30
translation_matrix = np.float32([[1, 0, Tx], [0, 1, Ty]])
translated_image = apply_transformation(image, translation_matrix)

# Rotation
rows, cols = image.shape[:2]  # Get the dimensions of the image
angle = 30
rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
rotated_image = apply_transformation(image, rotation_matrix)

# Scaling
Sx, Sy = 1.5, 1.5
scaling_matrix = np.float32([[Sx, 0, 0], [0, Sy, 0]])
scaled_image = apply_transformation(image, scaling_matrix)

# Shearing
Shx, Shy = 0.2, 0.3
shear_matrix = np.float32([[1, Shx, 0], [Shy, 1, 0]])
sheared_image = apply_transformation(image, shear_matrix)

# Display results
fig, axes = plt.subplots(1, 4, figsize=(15, 5))
axes[0].imshow(translated_image, cmap='gray'); axes[0].set_title('Translated')
axes[1].imshow(rotated_image, cmap='gray'); axes[1].set_title('Rotated')
axes[2].imshow(scaled_image, cmap='gray'); axes[2].set_title('Scaled')
axes[3].imshow(sheared_image, cmap='gray'); axes[3].set_title('Sheared')
plt.show()


# In[26]:


import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to apply transformation (translation, rotation, scaling, shearing)
def apply_transformation(image, matrix):
    rows, cols = image.shape[:2]
    transformed_image = cv2.warpAffine(image, matrix, (cols, rows))
    return transformed_image

# Load the image in grayscale
image = cv2.imread(r"C:\Users\student\Desktop\lion.jpg", 0)

# Task 1: Translation - Apply translation with different values (Tx, Ty)
Tx, Ty = 50, 30
translation_matrix = np.float32([[1, 0, Tx], [0, 1, Ty]])
translated_image = apply_transformation(image, translation_matrix)

Tx2, Ty2 = 100, 50  # Different translation values for comparison
translation_matrix2 = np.float32([[1, 0, Tx2], [0, 1, Ty2]])
translated_image2 = apply_transformation(image, translation_matrix2)

# Task 2: Rotation - Apply rotation by 30°, 60°, and 90°
rows, cols = image.shape[:2]
rotation_matrix_30 = cv2.getRotationMatrix2D((cols/2, rows/2), 30, 1)
rotation_matrix_60 = cv2.getRotationMatrix2D((cols/2, rows/2), 60, 1)
rotation_matrix_90 = cv2.getRotationMatrix2D((cols/2, rows/2), 90, 1)

rotated_image_30 = apply_transformation(image, rotation_matrix_30)
rotated_image_60 = apply_transformation(image, rotation_matrix_60)
rotated_image_90 = apply_transformation(image, rotation_matrix_90)

# Task 3: Scaling - Apply scaling with different scale factors
Sx, Sy = 1.5, 1.5
scaling_matrix = np.float32([[Sx, 0, 0], [0, Sy, 0]])
scaled_image = apply_transformation(image, scaling_matrix)

Sx2, Sy2 = 2.0, 2.0  # Larger scale factors
scaling_matrix2 = np.float32([[Sx2, 0, 0], [0, Sy2, 0]])
scaled_image2 = apply_transformation(image, scaling_matrix2)

# Task 4: Shearing - Apply shearing with different shear factors
Shx, Shy = 0.2, 0.3
shear_matrix = np.float32([[1, Shx, 0], [Shy, 1, 0]])
sheared_image = apply_transformation(image, shear_matrix)

Shx2, Shy2 = 0.5, 0.5  # Larger shear factors
shear_matrix2 = np.float32([[1, Shx2, 0], [Shy2, 1, 0]])
sheared_image2 = apply_transformation(image, shear_matrix2)

# Task 5: Display and Save Images

# Create a subplot to visualize the results
fig, axes = plt.subplots(3, 4, figsize=(15, 10))

# Task 1: Display Translations
axes[0, 0].imshow(translated_image, cmap='gray')
axes[0, 0].set_title(f'Translated Tx={Tx}, Ty={Ty}')
axes[0, 1].imshow(translated_image2, cmap='gray')
axes[0, 1].set_title(f'Translated Tx={Tx2}, Ty={Ty2}')

# Task 2: Display Rotations
axes[0, 2].imshow(rotated_image_30, cmap='gray')
axes[0, 2].set_title('Rotated 30°')
axes[0, 3].imshow(rotated_image_60, cmap='gray')
axes[0, 3].set_title('Rotated 60°')
axes[1, 3].imshow(rotated_image_90, cmap='gray')
axes[1, 3].set_title('Rotated 90°')

# Task 3: Display Scaled Images
axes[1, 0].imshow(scaled_image, cmap='gray')
axes[1, 0].set_title(f'Scaled Sx={Sx}, Sy={Sy}')
axes[1, 1].imshow(scaled_image2, cmap='gray')
axes[1, 1].set_title(f'Scaled Sx={Sx2}, Sy={Sy2}')

# Task 4: Display Sheared Images
axes[2, 0].imshow(sheared_image, cmap='gray')
axes[2, 0].set_title(f'Sheared Shx={Shx}, Shy={Shy}')
axes[2, 1].imshow(sheared_image2, cmap='gray')
axes[2, 1].set_title(f'Sheared Shx={Shx2}, Shy={Shy2}')

# Hide axes and show the plot
for ax in axes.ravel():
    ax.axis('off')

plt.tight_layout()
plt.show()

# Save images to disk
cv2.imwrite(r"C:\Users\student\Desktop\translated_image1.jpg", translated_image)
cv2.imwrite(r"C:\Users\student\Desktop\translated_image2.jpg", translated_image2)
cv2.imwrite(r"C:\Users\student\Desktop\rotated_image_30.jpg", rotated_image_30)
cv2.imwrite(r"C:\Users\student\Desktop\rotated_image_60.jpg", rotated_image_60)
cv2.imwrite(r"C:\Users\student\Desktop\rotated_image_90.jpg", rotated_image_90)
cv2.imwrite(r"C:\Users\student\Desktop\scaled_image1.jpg", scaled_image)
cv2.imwrite(r"C:\Users\student\Desktop\scaled_image2.jpg", scaled_image2)
cv2.imwrite(r"C:\Users\student\Desktop\sheared_image1.jpg", sheared_image)
cv2.imwrite(r"C:\Users\student\Desktop\sheared_image2.jpg", sheared_image2)

