from keras import layers
from keras import models
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import numpy as np
from keras.applications import InceptionV3

conv_base = InceptionV3(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))
//conv_base.summary()

batch_size = 20
def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='categorical')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            # Note that since generators yield data indefinitely in a loop,
            # we must `break` after every image has been seen once.
            break
    return features, labels

datagen = ImageDataGenerator(
    rescale=1. /255)

train_dir = 'Datasets/train'
validation_dir = 'Datasets/validation'
test_dir = 'Datasets/test'

train_features, train_labels = extract_features(train_dir, 2593)
validation_features, validation_labels = extract_features(validation_dir, 865)
test_features, test_labels = extract_features(test_dir, 865)

train_features = np.reshape(train_features, (2593, 3 * 3 * 2048))
validation_features = np.reshape(validation_features, (865, 3 * 3 * 2048))
test_features = np.reshape(test_features, (865, 3 * 3 * 2048))

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=3 * 3 * 2048))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(5, activation='softmax'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='categorical_crossentropy',
              metrics=['acc'])

history = model.fit(train_features, train_labels,
                    epochs=3,
                    batch_size=20,
                    validation_data=(validation_features, validation_labels))


test_loss, test_acc = model.evaluate(test_features, test_labels, use_multiprocessing=True)
print(test_loss,test_acc)

