from keras import layers
from keras import models
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
if __name__ ==  '__main__':

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(5, activation='softmax'))

    model.compile(loss='categorical_crossentropy',optimizer=optimizers.RMSprop(),metrics=['acc'])

    datagen = ImageDataGenerator(
        rescale=1. /255,
          rotation_range=40,
          width_shift_range=0.2,
          height_shift_range=0.2,
          shear_range=0.2,
          zoom_range=0.2,
          horizontal_flip=True,
          fill_mode='nearest')


    # All images will be rescaled by 1./255
    train_datagen = datagen
    val_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_dir = 'Datasets/train'
    validation_dir = 'Datasets/validation'
    test_dir = 'Datasets/test'
    batch_size = 20
    train_generator = train_datagen.flow_from_directory(
            # This is the target directory
            train_dir,
            # All images will be resized to 150x150
            target_size=(150, 150),
            batch_size=batch_size,
            # Since we use binary_crossentropy loss, we need binary labels
            class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
            validation_dir,
            target_size=(150, 150),
            batch_size=batch_size,
            class_mode='categorical')

    test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(150, 150),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False)

    noModel = True
    if noModel:
        history = model.fit_generator(
              train_generator,
              steps_per_epoch=130,
              epochs=8,
              validation_data=validation_generator,
              validation_steps=44)
        model.save('flowers2.h5')
    else:
        from keras.models import load_model
        model = load_model('flowers2.h5')


    test_loss, test_acc = model.evaluate_generator(test_generator)

    print(test_loss, test_acc)

    Y_pred = model.predict_generator(test_generator, verbose=1)
    #print(Y_pred)
    y_pred = np.argmax(Y_pred, axis=1)
    #print(test_generator.classes)
    #print(y_pred)
    print('Confusion Matrix')
    print(confusion_matrix(test_generator.classes, y_pred))
    print('Classification Report')
    target_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
    print(classification_report(validation_generator.classes, y_pred, target_names=target_names))