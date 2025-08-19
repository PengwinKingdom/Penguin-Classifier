#Libraries
import os,json
import numpy as np
from collections import Counter
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint

def main():
    # Dataset and preprocessing parameters
    dataset_path='Penguin_species'
    img_size=(224,224)
    batch_size=32
    seed=42
    
    #Image preprocessing and data augmentation
    train_datagen=ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.15,
        zoom_range=0.2,
        brightness_range=(0.9,1.1),
        horizontal_flip=True
    )
    
    val_datagen=ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    #Load training images from 'Penguin_species' folder
    train_generator=train_datagen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        color_mode='rgb',
        shuffle=True,
        seed=seed
    )
    
    #Load validation images from 'Penguin_species' folder
    val_generator=val_datagen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        color_mode='rgb',
        shuffle=False
    )
    
    num_classes=train_generator.num_classes
    
    os.makedirs('model',exist_ok=True)
    with open('model/class_indices.json','w') as f:
        json.dump(train_generator.class_indices,f,indent=2)
    
    base_model=MobileNetV2(input_shape=img_size+(3,),include_top=False,weights='imagenet')
    base_model.trainable=False
    
    x=base_model.output
    x=GlobalAveragePooling2D()(x)
    x=Dropout(0.2)(x)
    x=Dense(128,activation='relu')(x)
    output=Dense(num_classes,activation='softmax')(x) #18 penguin species
    
    model=Model(inputs=base_model.input,outputs=output)
    model.compile(optimizer=Adam(learning_rate=1e-3),loss='categorical_crossentropy',metrics=['accuracy'])
    
    #Train classification head
    early_stop=EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
    reduce_lr=ReduceLROnPlateau(monitor='val_loss',factor=0.3,patience=3,min_lr=1e-6,verbose=1)
    ckpt_head=ModelCheckpoint('model/best_head.h5',monitor='val_loss',save_best_only=True,verbose=1)
    
    counts=Counter(train_generator.classes)
    total=sum(counts.values())
    class_weights={i:total/(num_classes*counts[i]) for i in range(num_classes)}
    
    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=20,
        callbacks=[early_stop, reduce_lr, ckpt_head],
        class_weight=class_weights,
        verbose=1
    )

    # Fine-tuning
    for layer in base_model.layers[-60:]:
        layer.trainable = True
        
    model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    ckpt_ft = ModelCheckpoint('model/best_finetuned.h5', monitor='val_loss', save_best_only=True, verbose=1)

    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=30,
        callbacks=[early_stop, reduce_lr, ckpt_ft],
        class_weight=class_weights,
        verbose=1
    )

    #Save trained model to 'model' folder
    model.save('model/penguin_classifier.h5')
    print('Saved model to model/penguin_classifier.h5')
    
if __name__=='__main__':
    main()