import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import gridspec

#%% 
# Playing with the kernel
# Kernel: Matrix in 1D, 2D or 3D depending of the data, with which one will be made the filter
# Strides: Every how many pixels will be applied the filter
# Padding: What happend with the bordes. "VALID" keep the kernel into the image, "SAME" add ceros on  the borders 
# Pooling_type: With kind of pool will be applied "MAX" or "AVG"

plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('image', cmap='magma')

# Read image
image_path = './input/computer-vision-resources/car_illus.jpg'
image = tf.io.read_file(image_path)
image = tf.io.decode_jpeg(image, channels=1)
image = tf.image.resize(image, size=[400, 400])

# Embossing kernel
kernel = tf.constant([
    [-2, -1, 0],
    [-1, 1, 1],
    [0, 1, 2],
])

# Reformat for batch compatibility.
image = tf.image.convert_image_dtype(image, dtype=tf.float32)
image = tf.expand_dims(image, axis=0)
kernel = tf.reshape(kernel, [*kernel.shape, 1, 1])
kernel = tf.cast(kernel, dtype=tf.float32)

# Setting the filter
image_filter = tf.nn.conv2d(
    input=image,
    filters=kernel,
    strides=1,
    padding='VALID',
)

# Setting the ReLu
image_detect = tf.nn.relu(image_filter)

# Setting the Pool (could be "MAX" and "AVG")
image_condense = tf.nn.pool(
    input=image_detect,
    window_shape=[2,2],
    pooling_type = "MAX",
    strides = (2,2),
    padding="SAME"
)

# Show what we have so far
plt.figure(figsize=(12, 6))
plt.subplot(141)
plt.imshow(tf.squeeze(image), cmap='gray')
plt.axis('off')
plt.title('Input')
plt.subplot(142)
plt.imshow(tf.squeeze(image_filter))
plt.axis('off')
plt.title('Filter')
plt.subplot(143)
plt.imshow(tf.squeeze(image_detect))
plt.axis('off')
plt.title('Detect')
plt.subplot(144)
plt.imshow(tf.squeeze(image_condense))
plt.axis('off')
plt.title('Condense')


