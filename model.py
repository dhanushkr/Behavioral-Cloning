import numpy as np
import cv2
import csv

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers import Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

def get_samples(path):
    samples = []
    with open(path+'/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        next(reader,None)
        for line in reader:
            samples.append(line)
    return samples

def get_image_data(samples):
    center = []
    left = []
    right = []
    measurement = []

    for sample in samples:
        center.append('./data/'+sample[0].strip())
        left.append('./data/'+sample[1].strip())
        right.append('./data/'+sample[2].strip())
        measurement.append(float(sample[3]))
    return center,left,right,measurement

def combine_images(samples,correction = 0.2):
    center,left,right,measurement = get_image_data(samples)
    measurements = []
    image_paths = []
    
    image_paths.extend(center)
    image_paths.extend(left)
    image_paths.extend(right)
 
    measurements.extend(measurement)
    measurements.extend([x+correction for x in measurement])
    measurements.extend([x-correction for x in measurement])
    return (image_paths,measurements)

def generator(samples,batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0,num_samples,batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for img_path, measurement in batch_samples:
                image  = cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB)
                angle = float(measurement)                
                images.append(image)
                angles.append(angle)                
                images.append(cv2.flip(image,1))
                angles.append(angle*-1.0)
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

           
samples = get_samples('./data/')
image_paths, measurement = combine_images(samples)
samples = list(zip(image_paths, measurement))

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

train_generator = generator(train_samples)
validation_generator = generator(validation_samples)

def NVIDIAModel():
  model = Sequential()
  model.add(Lambda(lambda x: (x/255.0) - 0.5,input_shape=(160,320,3)))
  model.add(Cropping2D(cropping=((50,20),(0,0))))
  model.add(Conv2D(24,kernel_size=(5,5),strides=(2,2),activation='relu'))
  model.add(Conv2D(36,kernel_size=(5,5),strides=(2,2),activation='relu'))
  model.add(Conv2D(48,kernel_size=(5,5),strides=(2,2),activation='relu'))
  model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
  model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
  model.add(Flatten())
  model.add(Dense(100))
  model.add(Dense(50))
  model.add(Dense(10))
  model.add(Dense(1))
  return model

model = NVIDIAModel()
model.compile(loss='mse',optimizer = 'adam')
history_object = model.fit_generator(train_generator, 
            steps_per_epoch=len(train_samples),
            validation_data=validation_generator, 
            validation_steps=len(validation_samples),epochs=3,verbose=1)

model.save('model.h5')
print("Done")