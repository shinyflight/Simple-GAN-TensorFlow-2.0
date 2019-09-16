# !pip install -q tensorflow-gpu==2.0.0-rc1
import tensorflow as tf
import matplotlib.pyplot as plt

batch_size = 100
epochs = 10000
z_dim = 20

# Noise for visualization
z_vis = tf.random.normal([10, z_dim])

# Load data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255.0
x_iter = iter(tf.data.Dataset.from_tensor_slices(x_train).shuffle(4 * batch_size).batch(batch_size).repeat())

# Generator
G = tf.keras.models.Sequential([
  tf.keras.layers.Dense(28*28 // 2, input_shape = (z_dim,), activation='relu'),
  tf.keras.layers.Dense(28*28, activation='sigmoid'),
  tf.keras.layers.Reshape((28, 28))])

# Discriminator
D = tf.keras.models.Sequential([
 tf.keras.layers.Flatten(input_shape=(28, 28)),
 tf.keras.layers.Dense(28*28 // 2, activation='relu'),
 tf.keras.layers.Dense(1)])

# Loss functions
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits = True)
def G_loss(D, x_fake):
  return cross_entropy(tf.ones_like(D(x_fake)), D(x_fake))
def D_loss(D, x_real, x_fake):
  return cross_entropy(tf.ones_like(D(x_real)), D(x_real)) + cross_entropy(tf.zeros_like(D(x_fake)), D(x_fake))

# Optimizers
G_opt = tf.keras.optimizers.Adam(1e-4)
D_opt = tf.keras.optimizers.Adam(1e-4)

# Train
for epoch in range(epochs):
  z_mb = tf.random.normal([batch_size, z_dim])
  x_real = next(x_iter)
  # Record operations
  with tf.GradientTape() as G_tape, tf.GradientTape() as D_tape:  
    x_fake = G(z_mb)
    G_loss_curr = G_loss(D, x_fake)
    D_loss_curr = D_loss(D, x_real, x_fake)
  # Gradients
  G_grad = G_tape.gradient(G_loss_curr, G.trainable_variables)
  D_grad = D_tape.gradient(D_loss_curr, D.trainable_variables)
  # Apply gradients
  G_opt.apply_gradients(zip(G_grad, G.trainable_variables))
  D_opt.apply_gradients(zip(D_grad, D.trainable_variables))
  
  if epoch % 100 == 0:
    # Print results
    print('epoch: {}; G_loss: {:.6f}; D_loss: {:.6f}'.format(epoch+1, G_loss_curr, D_loss_curr))
    # Plot generated images
    for i in range(10):
      plt.subplot(2, 5, i+1)
      plt.imshow(G(z_vis)[i,:,:]*255.0)
      plt.axis('off')
    plt.show()
