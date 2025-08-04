#Libraries
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

def main():
    dataset_path='Penguin_species'
    img_size=(224,224) # Input size required by MobileNetV2
    batch_size=32
    
    
    #Image preprocessing and data augmentation
    train_data=ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True
    )
    
    #Load training images from 'Penguin_species' folder
    train_generator=train_data.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        color_mode='rgb'
    )
    
    #Load validation images from 'Penguin_species' folder
    val_generator=train_data.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        color_mode='rgb'
    )
    
    base_model=MobileNetV2(input_shape=img_size+(3,),include_top=False,weights='imagenet')
    base_model.trainable=False
    
    x=base_model.output
    x=GlobalAveragePooling2D()(x)
    x=Dense(64,activation='relu')(x)
    output=Dense(18,activation='softmax')(x) #18 penguin species
    
    model=Model(inputs=base_model.input,outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])
    
    
    #Training the model
    early_stop=EarlyStopping(patience=3,restore_best_weights=True)
    
    history=model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=30,
        callbacks=[early_stop]
    )
    
    #Save trained model to 'model' folder
    os.makedirs('model',exist_ok=True)
    model.save('model/penguin_classifier.h5')
    
if __name__=='__main__':
    main()