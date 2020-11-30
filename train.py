import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from IPython.display import clear_output
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image

from tensorflow_examples.models.pix2pix import pix2pix

import matplotlib
matplotlib.use('qt5agg')

classColors = {
  1: "#402020", # road (all parts, anywhere nobody would look at you funny for driving)
  2: "#ff0000", # lane markings (don't include non lane markings like turn arrows and crosswalks)
  3: "#808060", # undrivable
  4: "#00ff66", # movable (vehicles and people/animals)
  5: "#cc00ff", # my car (and anything inside it, including wires, mounts, etc. No reflections)
}

colors = [
  (64, 32, 32), # road (all parts, anywhere nobody would look at you funny for driving)
  (255, 0, 0), # lane markings (don't include non lane markings like turn arrows and crosswalks)
  (128, 128, 96), # undrivable
  (0, 255, 102), # movable (vehicles and people/animals)
  (204, 0, 255) # my car (and anything inside it, including wires, mounts, etc. No reflections)
]

def filename_to_trainingdp(filename):
  return {
    "image": tf.image.decode_image(tf.io.read_file("imgs/" + filename), channels=3, expand_animations=False),
    "segmentation_mask": tf.image.decode_image(tf.io.read_file("masks/" + filename), channels=3, expand_animations=False)
    #"segmentation_mask": tf.keras.preprocessing.image.img_to_array(tf.keras.preprocessing.image.load_img("masks/" + filename, grayscale=False, color_mode='rgb'))
          }

def normalize(input_image, input_mask):
  input_image = tf.cast(input_image, tf.float32) / 255.0
  input_mask = tf.cast(input_mask, tf.uint8)
  

  #input_mask -= 1
  return input_image, input_mask

@tf.function
def load_image_train(datapoint):
    # print(datapoint)
    input_image = tf.image.resize(datapoint['image'], (128, 128))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))
  
    one_hot_map = []
    for color in colors:
      class_map = tf.reduce_all(tf.equal(input_mask, color), axis=-1)
      one_hot_map.append(class_map)

    one_hot_map = tf.stack(one_hot_map, axis=-1)
    one_hot_map = tf.cast(one_hot_map, tf.float32)
    input_mask = tf.argmax(one_hot_map, axis=-1)

    
    """
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)
    """
    
    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

def load_image_test(datapoint):
  input_image = tf.image.resize(datapoint['image'], (128, 128))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

  input_image, input_mask = normalize(input_image, input_mask)

  return input_image, input_mask


file1 = open('files_trainable', 'r') 
filenames = list(map(lambda x: os.path.basename(x.rstrip()), file1.readlines()))
file1.close()

dataset = tf.data.Dataset.from_tensor_slices(list(filenames))
dataset = dataset.map(filename_to_trainingdp, num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataset = dataset.map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)

tf.print(dataset)

exit

TRAIN_LENGTH = len(filenames) # images.len()
BATCH_SIZE = 64
BUFFER_SIZE = 9000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

train_dataset = dataset.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    tf.print(display_list[i].shape)
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    if display_list[i].shape == (128,128):
      plt.imshow(display_list[i])
    else:
      plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()

for image, mask in dataset.take(1):
  sample_image, sample_mask = image, mask
#display([sample_image, sample_mask])

base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

# Use the activations of these layers
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]
layers = [base_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

down_stack.trainable = False

up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]

def unet_model(output_channels):
  inputs = tf.keras.layers.Input(shape=[128, 128, 3])
  x = inputs

  # Downsampling through the model
  skips = down_stack(x)
  x = skips[-1]
  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = tf.keras.layers.Concatenate()
    x = concat([x, skip])

  # This is the last layer of the model
  last = tf.keras.layers.Conv2DTranspose(filters=output_channels, kernel_size=3, strides=2, padding='same')  #64x64 -> 128x128

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

OUTPUT_CHANNELS = len(colors)

model = unet_model(OUTPUT_CHANNELS)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

tf.keras.utils.plot_model(model, show_shapes=True)

def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]

def show_predictions(dataset=None, num=1):
  if dataset:
    for image, mask in dataset.take(num):
      pred_mask = model.predict(image)
      display([image[0], mask[0], create_mask(pred_mask)])
  else:
    display([sample_image, sample_mask,
             create_mask(model.predict(sample_image[tf.newaxis, ...]))])

# show_predictions()

class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    clear_output(wait=True)
    # show_predictions()
    print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

if os.path.isfile("weights.h5"):
	model.load_weights("weights.h5")

show_predictions()

EPOCHS = 20
VAL_SUBSPLITS = 5
VALIDATION_STEPS = 2 # info.splits['test'].num_examples//BATCH_SIZE//VAL_SUBSPLITS

model_history = model.fit(train_dataset, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=train_dataset,
                          callbacks=[DisplayCallback()])

model.save_weights('weights.h5')

show_predictions(train_dataset, 3)
