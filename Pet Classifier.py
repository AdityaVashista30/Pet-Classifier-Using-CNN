

import tensorflow
from keras_preprocessing.image import ImageDataGenerator

#PART 1: IMAGE PREPROCESSING

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
"""### Generating images for the Test set"""

test_datagen = ImageDataGenerator(rescale = 1./255)
"""### Creating the Training set"""

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (128, 128),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

"""### Creating the Test set"""

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (128, 128),
                                            batch_size = 32,
                                            class_mode = 'binary')


#PART 2: Creating CNN Classifiaction Model
cnn=tensorflow.keras.models.Sequential()

#ADDING FIRST LAYER
#STEP 1: convolution
cnn.add(tensorflow.keras.layers.Conv2D(filters=64,kernel_size=3,input_shape=[128, 128, 3],padding="same", activation="relu"))
#STEP 1: convolution
cnn.add(tensorflow.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

#ADDING 2nd LAYER
cnn.add(tensorflow.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
cnn.add(tensorflow.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

#STEP 3: FLATTENING (LAYER FINAL)
cnn.add(tensorflow.keras.layers.Flatten())

#STEP 4: FULL CONNECTION
cnn.add(tensorflow.keras.layers.Dense(units=128, activation='relu'))

"""### Step 5 - Output Layer"""
cnn.add(tensorflow.keras.layers.Dense(units=1, activation='sigmoid'))

### Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#FITTING AND TESTING
cnn.fit(training_set,
                  steps_per_epoch = (8048/32),
                  epochs = 32,
                  validation_data = test_set,
                  validation_steps = (2000/32))

#PREDICTING 
import numpy as np
from keras_preprocessing import image
testImage1=image.load_img('dataset/single_prediction/cat_or_dog_1.jpg',target_size=(128,128))
testImage2=image.load_img('dataset/single_prediction/cat_or_dog_2.jpg',target_size=(128,128))
testImage1=image.img_to_array(testImage1)
testImage2=image.img_to_array(testImage2)
testImage1=np.expand_dims(testImage1,0)
testImage2=np.expand_dims(testImage2,0)
results=[cnn.predict(testImage1),cnn.predict(testImage2)]
for i in range(len(results)):
    if results[i][0,0]==1:
        print("image ",i+1," is dog")
    else:
        print("image ",i+1," is cat")
        
#SAVING MODEL
cnn.save("pet_classifier.h5")
#DELETING MODEL
del cnn
#OPENING SAVED MODEL
from tensorflow.keras.models import load_model
classifier=load_model("pet_classifier.h5")

test3=image.load_img('dataset/single_prediction/unkown.jpg',target_size=(128,128))
test3=image.img_to_array(test3)
test3=np.expand_dims(test3,0)
if(classifier.predict(test3)[0][0]==1):
    print("it is dog")
else:
    print("it is cat")
