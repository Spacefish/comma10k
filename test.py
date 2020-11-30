import tensorflow as tf

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