#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/JaishreeSha/Falldetection/blob/main/Falldetction_CNN.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


import os
train_dir = '/content/drive/MyDrive/Dataset/train'
validation_dir ='/content/drive/MyDrive/Dataset/test'


# In[ ]:


from tensorflow.keras import layers
from tensorflow.keras import Model


# In[ ]:


img_input = layers.Input(shape=(150, 150, 3))
x = layers.Conv2D(16, 3, activation='relu')(img_input)
x = layers.MaxPooling2D(2)(x)
x = layers.Conv2D(32, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)
x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)
x = layers.Flatten()(x)
x = layers.Dense(512, activation='relu')(x)
output = layers.Dense(1, activation='sigmoid')(x)
model = Model(img_input, output)


# In[ ]:


# Flatten feature map to a 1-dim tensor so we can add fully connected layers
x = layers.Flatten()(x)

# Create a fully connected layer with ReLU activation and 512 hidden units
x = layers.Dense(512, activation='relu')(x)

# Create output layer with a single node and sigmoid activation
output = layers.Dense(1, activation='sigmoid')(x)

# Create model:
# input = input feature map
# output = input feature map + stacked convolution/maxpooling layers + fully
# connected layer + sigmoid output layer
model = Model(img_input, output)


# In[ ]:


model.summary()


# In[ ]:


from tensorflow.keras.optimizers import RMSprop

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['acc'])


# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
    train_dir,  # This is the source directory for training images
    target_size=(150, 150),  # All images will be resized to 150x150
    batch_size=20,
    # Since we use binary_crossentropy loss, we need binary labels
    class_mode='binary')

# Flow validation images in batches of 20 using val_datagen generator
validation_generator = val_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')


# In[ ]:


history=model.fit(train_generator,epochs = 12,validation_data = validation_generator)


# In[ ]:


# Saving Model
model.save(filepath='/content/drive/MyDrive/Dataset/fall_detection_model.h5', overwrite=True)


# In[ ]:


from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
def load_image(filename):
    # load the image
    img = load_img(filename, target_size=(150,150))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 3 channels
    img = img.reshape(1,150,150, 3)
    # center pixel data
    img = img.astype('float32')
    img = img - [123.68, 116.779, 103.939]
    return img
img = load_image('/content/drive/MyDrive/Dataset/test/Fall/fall001.jpg')
# predict the class
result = model.predict(img)
if(result[0]==1):
    print("Fall")
else:
    print("Not Fall")


# In[ ]:


import pandas as pd
import numpy as np
from keras.preprocessing import image
y_actual, y_test = [],[]
for i in os.listdir("/content/drive/MyDrive/Dataset/train/Notfall/"):
    img=image.load_img("/content/drive/MyDrive/Dataset/train/Notfall/"+i,target_size=(150,150))
    img=image.img_to_array(img)
    img=np.expand_dims(img,axis=0)
    pred=model.predict(img)
    y_test.append(int(pred[0,0]))
    y_actual.append(1)

for i in os.listdir("/content/drive/MyDrive/Dataset/train/Fall/"):
    img=image.load_img("/content/drive/MyDrive/Dataset/train/Fall/"+i,target_size=(150,150))
    img=image.img_to_array(img)
    img=np.expand_dims(img,axis=0)
    pred=model.predict(img)
    y_test.append(int(pred[0,0]))
    y_actual.append(0)


# In[ ]:


from sklearn import metrics

expected_y  = y_actual
y_pred= y_test

# summarize the fit of the model
print(); print(metrics.classification_report(expected_y,y_pred))
print(); print(metrics.confusion_matrix(expected_y, y_pred))
print("Accuracy:  ",metrics.accuracy_score(expected_y, y_pred))


# In[ ]:


# calculate accuracy
y_test=y_actual
from sklearn import metrics
print("ACCURACY:")
print(metrics.accuracy_score(y_test, y_pred))
confusion = metrics.confusion_matrix(y_test, y_pred)
print(confusion)
#[row, column]
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]
print("sensitivity")
print(metrics.recall_score(y_test, y_pred))
print("True Positive Rate")
specificity = TN / (TN + FP)
print(specificity)
false_positive_rate = FP / float(TN + FP)
print("false_positive_rate")
print(false_positive_rate)
print("precision")
print(metrics.precision_score(y_test, y_pred))
print("ROC_AUC_SCORE")
print(metrics.roc_auc_score(y_test, y_pred))
from sklearn.metrics import f1_score
score = f1_score(y_test, y_pred, average='binary')
print('F-Measure: %.3f' % score)


# In[ ]:


import matplotlib.pyplot as plt
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC_CURVE for fall detection CNN model')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)


# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6,6))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted Fall', 'Predicted Not Fall'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual Fall', 'Actual Not Fall'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.title("Confusion matrix ")
plt.show()

