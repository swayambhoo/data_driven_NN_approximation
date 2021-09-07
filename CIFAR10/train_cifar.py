import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
import tensorflow.keras as keras
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

test_images = test_images.astype("float32")/255.0
test_labels = test_labels.astype("float32")

train_images = train_images.astype("float32")/255.0
train_labels = train_labels.astype("float32")

val_images = train_images[-10000:]
val_labels = train_labels[-10000:]
train_images = train_images[:-10000]
train_labels = train_labels[:-10000]


# random shuffling is very very important ...
np.random.seed(8)
ind = np.random.choice(train_images.shape[0], train_images.shape[0], replace=False)
train_images = train_images[ind, :]
train_labels = train_labels[ind]

#train_images = train_images[:1024]
#train_labels = train_labels[:1024]

#print(train_images.shape)

LenetModel = keras.Sequential(
    [
        keras.Input(shape=(32,32,3)),
        keras.layers.Conv2D(filters=6, kernel_size=(5,5), padding='valid', activation='relu', name="conv_1"),
        keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid', name="maxpool_1" ),
        keras.layers.Conv2D(filters=16, kernel_size=(5,5), padding='valid', activation='relu', name="conv_2"),
        keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid', name="maxpool_2" ),
        keras.layers.Flatten(name='flatten_1'),
        keras.layers.Dense(120, activation="relu", name="dense_1" ),
        keras.layers.Dense(84, activation="relu", name="dense_2" ),
        keras.layers.Dense(10, activation="softmax", name="dense_3")    
    ]

)

LenetModel.compile(optimizer=keras.optimizers.Adam(learning_rate = 1e-3), 
               loss=keras.losses.SparseCategoricalCrossentropy(), 
               metrics=[keras.metrics.SparseCategoricalAccuracy()])

LenetModel.fit(train_images,
    train_labels,
    batch_size=64,
    epochs=30,
    validation_data=(val_images, val_labels))

LenetModel.evaluate(test_images, test_labels, batch_size=128)

#LenetModel.save('Lenet_cifar10_Run1')


#Lenet_cifar20_Run2 is trained with 2048 samples.
#Lenet_cifar10_Run1 is trained with 1024 samples.
