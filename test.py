import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix

import numpy as np
import cv2

from IPython.display import clear_output
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('GTK3Agg')
from PIL import Image

cap = cv2.VideoCapture('test/VID_20150824_200224.mp4')
# cap = cv2.VideoCapture('test/dashcam.mp4')

size = 128*3

colors = [
  (64, 32, 32), # road (all parts, anywhere nobody would look at you funny for driving)
  (255, 0, 0), # lane markings (don't include non lane markings like turn arrows and crosswalks)
  (128, 128, 96), # undrivable
  (0, 255, 102), # movable (vehicles and people/animals)
  (204, 0, 255) # my car (and anything inside it, including wires, mounts, etc. No reflections)
]

base_model = tf.keras.applications.MobileNetV2(input_shape=[size, size, 3], include_top=False, alpha=1.4)

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
  inputs = tf.keras.layers.Input(shape=[size, size, 3])
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

model.load_weights("weights.h5")

def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]

palette = {i: tf.constant(color, dtype='int64') for i, color in enumerate(
  ((64, 32, 32), # road (all parts, anywhere nobody would look at you funny for driving)
  (255, 0, 0), # lane markings (don't include non lane markings like turn arrows and crosswalks)
  (128, 128, 96), # undrivable
  (0, 255, 102), # movable (vehicles and people/animals)
  (204, 0, 255))
)}

palette = tf.constant(((64, 32, 32), # road (all parts, anywhere nobody would look at you funny for driving)
  (255, 0, 0), # lane markings (don't include non lane markings like turn arrows and crosswalks)
  (128, 128, 96), # undrivable
  (0, 255, 102), # movable (vehicles and people/animals)
  (204, 0, 255)), dtype='uint8')

print(palette)
#palette = tf.constant(palette, dtype=tf.uint8)

c = 0

while(cap.isOpened()):
    ret, frame = cap.read()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #gray = cv2.cvtColor(frame, cv2.COLOR_RGB)

    imagePayload = tf.keras.preprocessing.image.img_to_array(frame)
    
    image = tf.image.resize(imagePayload, (size,size))
    image = image / 255.0
    #tf.print(image)

    prediction = model.predict(image[tf.newaxis, ...])
    predicted_mask = create_mask(prediction)
    
    W, H, _ = predicted_mask.get_shape()   # this returns batch_size, 128, 128, 5

    class_indexes = tf.reshape(predicted_mask, [-1])
    color_image = tf.gather(palette, class_indexes)
    color_image = tf.reshape(color_image, [-1, W, H, 3])

    realImg = tf.image.convert_image_dtype(image, dtype=tf.uint8)
    realImg = tf.keras.preprocessing.image.img_to_array(realImg)
    maskImg = tf.keras.preprocessing.image.img_to_array(color_image[0])
  
    realImg = Image.fromarray(np.uint8(realImg))
    maskImg = Image.fromarray(np.uint8(maskImg))

    outImg = Image.new('RGBA', (size*2,size), (255,0,0,255))
    outImg.paste(realImg, (0,0))
    outImg.paste(maskImg, (size,0))

    outImg.save('./out/' + str(c).zfill(5) + ".png", "PNG")

    #tf.keras.preprocessing.image.save_img('./out/M' + str(c).zfill(5) + '.png', color_image[0]);
    #tf.keras.preprocessing.image.save_img('./out/F' + str(c).zfill(5) + '.png', image);
    #mask = Image.fromarray(predicted_mask)

    #float_image_tensor = tf.image.convert_image_dtype(image,tf.float32)
    if False:
      plt.figure(figsize=(15, 15))
      plt.subplot(1, 2, 1)
      plt.imshow(image)
      plt.subplot(1, 2, 2)
      plt.imshow(color_image[0])
      plt.show()

    c = c + 1

    #cv2.imshow('frame',bla)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break

cap.release()
cv2.destroyAllWindows()



imagePayload = tf.io.read_file("masks/0000_0085e9e41513078a_2018-08-19--13-26-08_11_864.png")
mask = tf.io.decode_png(imagePayload)

colors = [
  (64, 32, 32), # road (all parts, anywhere nobody would look at you funny for driving)
  (255, 0, 0), # lane markings (don't include non lane markings like turn arrows and crosswalks)
  (128, 128, 96), # undrivable
  (0, 255, 102), # movable (vehicles and people/animals)
  (204, 0, 255) # my car (and anything inside it, including wires, mounts, etc. No reflections)
]

one_hot_map = []
for color in colors:
    class_map = tf.reduce_all(tf.equal(mask, color), axis=-1)
    one_hot_map.append(class_map)

one_hot_map = tf.stack(one_hot_map, axis=-1)
one_hot_map = tf.cast(one_hot_map, tf.float32)

mask = tf.argmax(one_hot_map, axis=-1)

tf.print(mask)