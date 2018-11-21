from keras import layers
from keras import models
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers

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



# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1. / 255)
val_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_dir = 'Datasets/train'
validation_dir = 'Datasets/validation'
test_dir = 'Datasets/test'

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(150, 150),
        batch_size=10,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=10,
        class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=10,
        class_mode='categorical')

noModel = False;
if noModel:
    history = model.fit_generator(
          train_generator,
          steps_per_epoch=260,
          epochs=3,
          validation_data=validation_generator,
          validation_steps=86)
    model.save('flowers1.h5')
else:
    from keras.models import load_model
    model = load_model('flowers1.h5')
    test_loss, test_acc = model.evaluate_generator(test_generator, use_multiprocessing=True)
    print(test_loss,test_acc)

