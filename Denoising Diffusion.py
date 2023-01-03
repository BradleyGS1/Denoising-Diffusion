import math
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from keras import layers

tf.__version__
tf.config.list_physical_devices("GPU")

# -------------------------------------------------------------- #
# Hyperparameters

# Data
DATASET_NAME = "oxford_flowers102"
DATA_REPEATS = 5
IMAGE_SIZE = 64

# Model architecture
EMBEDDING_MAX_FREQ = 1000.0
WIDTHS = [32, 64, 96, 128]
BLOCK_DEPTH = 2

# Diffusion process
MIN_SIGNAL_RATE = 0.02
MAX_SIGNAL_RATE = 0.95

# Training
PLOT_DIFFUSION_STEPS = 25
NUM_EPOCHS = 5
BATCH_SIZE = 20
ETA = 1e-3

# -------------------------------------------------------------- #
# Data preparation

# Preprocess imgs function
def preprocess_imgs(imgs):
    # Centre crop
    height = tf.shape(imgs["image"])[0]
    width = tf.shape(imgs["image"])[1]

    crop_size = tf.minimum(height, width)
    imgs = tf.image.crop_to_bounding_box(
        imgs["image"],
        (height - crop_size) // 2,
        (width - crop_size) // 2,
        crop_size,
        crop_size
    )

    # Resize imgs to target image size
    imgs = tf.image.resize(
        imgs, 
        size = (IMAGE_SIZE, IMAGE_SIZE),
        antialias = True
    )

    # Return rescaled to [0, 1] imgs
    return tf.clip_by_value(imgs / 255.0, 0.0, 1.0)


# Define get data function with split argument for train and validation datasets
def get_data(split):
    return(
        tfds.load(DATASET_NAME, split = split, shuffle_files = True)
        .map(preprocess_imgs, num_parallel_calls = tf.data.AUTOTUNE)
        .cache()
        .repeat(DATA_REPEATS)
        .shuffle(5 * BATCH_SIZE)
        .batch(BATCH_SIZE, drop_remainder = True)
        .prefetch(buffer_size = tf.data.AUTOTUNE)
    )

train_data = get_data("train[:80%] + validation[:80%] + test[:80%]")
val_data = get_data("train[80%:] + validation[80%:] + test[80%:]")

# -------------------------------------------------------------- #
# Network Architecture

# Define a sinusoidal embedding to transform the noise variances into a tensor which
# the model will be highly sensitive to
def sinusoidal_embedding(x):
    frequencies = tf.exp(
        tf.linspace(
            tf.math.log(1.0),
            tf.math.log(EMBEDDING_MAX_FREQ),
            WIDTHS[0] // 2,
        )
    )
    angular_speeds = 2.0 * math.pi * frequencies
    embeddings = tf.concat(
        [tf.sin(angular_speeds * x), tf.cos(angular_speeds * x)], axis = 3
    )
    return embeddings

def ResidualBlock(width):
    def apply(x):
        input_width = x.shape[3]
        if input_width == width:
            residual = x
        else:
            residual = layers.Conv2D(width, kernel_size = 1)(x)

        x = layers.BatchNormalization(center = False, scale = False)(x)
        x = layers.Conv2D(
            width, kernel_size = 3, padding = "same", activation = keras.activations.swish
        )(x)
        x = layers.Conv2D(width, kernel_size = 3, padding = "same")(x)
        x = layers.Add()([x, residual])
        return x

    return apply


def DownBlock(width):
    def apply(x):
        x, skips = x
        for _ in range(BLOCK_DEPTH):
            x = ResidualBlock(width)(x)
            skips.append(x)

        x = layers.AveragePooling2D(pool_size = 2)(x)
        return x

    return apply


def UpBlock(width):
    def apply(x):
        x, skips = x
        x = layers.UpSampling2D(size = 2, interpolation = "bilinear")(x)
        for _ in range(BLOCK_DEPTH):
            x = layers.Concatenate()([x, skips.pop()])
            x = ResidualBlock(width)(x)   
        return x

    return apply

def get_network():
    noisy_imgs = keras.Input(shape = (IMAGE_SIZE, IMAGE_SIZE, 3))
    noise_variances = keras.Input(shape = (1, 1, 1))

    e = layers.Lambda(sinusoidal_embedding)(noise_variances)
    e = layers.UpSampling2D(size = IMAGE_SIZE, interpolation = "nearest")(e)

    x = layers.Conv2D(WIDTHS[0], kernel_size = 1)(noisy_imgs)
    x = layers.Concatenate()([x, e])

    skips = []
    for width in WIDTHS:
        x = DownBlock(width)([x, skips])

    for _ in range(BLOCK_DEPTH):
        x = ResidualBlock(WIDTHS[-1])(x)

    for width in reversed(WIDTHS):
        x = UpBlock(width)([x, skips])

    x = layers.Conv2D(3, kernel_size = 1, kernel_initializer = "zeros")(x)

    return keras.Model([noisy_imgs, noise_variances], x, name = "residual_unet")

network = get_network()
print(network.summary())
keras.utils.plot_model(
    network, 
    to_file = "Model Design.png", 
    show_shapes = True
)

# -------------------------------------------------------------- #
# Model Creation

class DiffusionModel(keras.Model):
    def __init__(self):
        super().__init__()

        # Initialise the normalisation layer and the neural network
        self.normalizer = layers.Normalization()
        self.network = get_network()

    def compile(self, **kwargs):
        super().compile(**kwargs)

        # Initialise the loss metrics trackers
        self.noise_loss_tracker = keras.metrics.Mean(name = "n_loss")
        self.image_loss_tracker = keras.metrics.Mean(name = "i_loss")

    @property
    def metrics(self):
        return [self.noise_loss_tracker, self.image_loss_tracker]

    def denormalize(self, imgs):
        # Converts pixel values back into range [0, 1]
        imgs = self.normalizer.mean + imgs * self.normalizer.variance**0.5
        return tf.clip_by_value(imgs, 0.0, 1.0)

    def diffusion_schedule(self, diffusion_times):
        start_angle = tf.acos(MAX_SIGNAL_RATE)
        end_angle = tf.acos(MIN_SIGNAL_RATE)

        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

        signal_rates = tf.cos(diffusion_angles)
        noise_rates = tf.sin(diffusion_angles)

        return noise_rates, signal_rates

    def denoise(self, noisy_imgs, noise_rates, signal_rates, training):
        # Predict the noise component and calculate the image component using it
        pred_noises = self.network([noisy_imgs, noise_rates**2], training = training)
        pred_imgs = (noisy_imgs - noise_rates * pred_noises) / signal_rates

        return pred_noises, pred_imgs

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        # Reverse diffusion = sampling
        num_imgs = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps

        # IMPORTANT:
        # At the first sampling step, the "noisy image" should theoretically be pure noise
        # But its signal rate is assumed to be nonzero (min_signal_rate)
        next_noisy_imgs = initial_noise
        for step in range(diffusion_steps):
            noisy_imgs = next_noisy_imgs

            # Separate the current noisy image from its components
            diffusion_times = tf.ones((num_imgs, 1, 1, 1)) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_imgs = self.denoise(
                noisy_imgs, noise_rates, signal_rates, training = False
            )

            # Find the predicted components using the next signal and noise rates
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times
            )
            next_noisy_imgs = (
                next_signal_rates * pred_imgs + next_noise_rates * pred_noises
            )
            # This new noisy image will be used in the next step

        return pred_imgs

    def generate(self, num_imgs, diffusion_steps):
        # noise -> imgs -> denormalized imgs
        initial_noise = tf.random.normal(shape = (num_imgs, IMAGE_SIZE, IMAGE_SIZE, 3))
        generated_imgs = self.reverse_diffusion(initial_noise, diffusion_steps)
        generated_imgs = self.denormalize(generated_imgs)
        return generated_imgs

    def train_step(self, imgs):
        # Normalize imgs to have standard deviation of 1, like the noises
        imgs = self.normalizer(imgs, training = True)
        noises = tf.random.normal(shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3))

        # Sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape = (BATCH_SIZE, 1, 1, 1), minval = 0.0, maxval = 1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # Mix the imgs with noises accordingly
        noisy_imgs = signal_rates * imgs + noise_rates * noises

        with tf.GradientTape() as tape:
            # Train the network to separate noisy imgs to their components
            pred_noises, pred_imgs = self.denoise(
                noisy_imgs, noise_rates, signal_rates, training = True
            )

            noise_loss = self.loss(noises, pred_noises)  # used for training
            image_loss = self.loss(imgs, pred_imgs)  # only used as metric

        gradients = tape.gradient(noise_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        self.noise_loss_tracker.update_state(noise_loss)
        self.image_loss_tracker.update_state(image_loss)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, imgs):
        # Normalize imgs to have standard deviation of 1, like the noises
        imgs = self.normalizer(imgs, training = False)
        noises = tf.random.normal(shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3))

        # Sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape = (BATCH_SIZE, 1, 1, 1), minval = 0.0, maxval = 1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # Mix the imgs with noises accordingly
        noisy_imgs = signal_rates * imgs + noise_rates * noises

        # Use the network to separate noisy imgs to their components
        pred_noises, pred_imgs = self.denoise(
            noisy_imgs, noise_rates, signal_rates, training = False
        )

        noise_loss = self.loss(noises, pred_noises)
        image_loss = self.loss(imgs, pred_imgs)

        self.image_loss_tracker.update_state(image_loss)
        self.noise_loss_tracker.update_state(noise_loss)

        return {m.name: m.result() for m in self.metrics}


    def plot_imgs(self, epoch = None, logs = None, num_rows = 3, num_cols = 6):
        # Plot random generated imgs for visual evaluation of generation quality
        generated_imgs = self.generate(
            num_imgs = num_rows * num_cols,
            diffusion_steps = PLOT_DIFFUSION_STEPS,
        )

        plt.figure(figsize = (num_cols * 2.0, num_rows * 2.0))
        for row in range(num_rows):
            for col in range(num_cols):

                index = row * num_cols + col
                plt.subplot(num_rows, num_cols, index + 1)
                plt.imshow(generated_imgs[index])
                plt.axis("off")
        plt.tight_layout()
        plt.show()


# --------------------------------------------------------- #
# Train model

# Create and compile the model
model = DiffusionModel()

# Pixelwise mean absolute error is used as loss
model.compile(
    optimizer = keras.optimizers.Adam(
        learning_rate = ETA
    ),
    loss = keras.losses.mean_absolute_error
)

# Calculate mean and variance of training dataset for normalization
model.normalizer.adapt(train_data)

# Save the model via callback
checkpoint_path = "checkpoints/diffusion_model"
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath = checkpoint_path,
    save_weights_only = True,
)

# Run training and plot generated imgs periodically
model.fit(
    train_data,
    epochs = NUM_EPOCHS,
    validation_data = val_data,
    callbacks = [
        keras.callbacks.LambdaCallback(on_epoch_end = model.plot_imgs),
        checkpoint_callback
    ]
)