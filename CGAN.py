"""
Conditional GAN (CGAN) Fine-Tuning Script using TensorFlow

This script fine-tunes a pretrained Conditional GAN by loading
an existing Generator and Discriminator and continuing adversarial
training on a new dataset with class labels.

Key characteristics:
- Uses pretrained CGAN models (Generator & Discriminator)
- Keeps architecture unchanged (true fine-tuning)
- Applies lower learning rates for stable adaptation
- Supports class-conditional image generation
- Designed for RGB images resized to 64x64

Use case:
Adapting an existing CGAN to a new but related dataset or domain
without training from scratch.
"""

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import pathlib

# =====================================================
# 1. Hyperparameters (MATCH PRETRAINED MODELS)
# =====================================================
latent_dim = 128
num_classes = 5
image_size = (64, 64)
channels = 3
batch_size = 32
epochs = 10

# =====================================================
# 2. Load Dataset (New / Target Dataset)
# =====================================================
data_dir = pathlib.Path("path/to/your/new_dataset")

dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels="inferred",
    label_mode="int",
    image_size=image_size,
    batch_size=batch_size
)

# Normalize to [-1, 1]
dataset = dataset.map(lambda x, y: ((x - 127.5) / 127.5, y))

# =====================================================
# 3. Load PRETRAINED Generator & Discriminator
# =====================================================
generator = tf.keras.models.load_model(
    "pretrained_generator.keras",
    compile=False
)

discriminator = tf.keras.models.load_model(
    "pretrained_discriminator.keras",
    compile=False
)

print("✅ Pretrained Generator and Discriminator loaded")

# =====================================================
# 4. (Optional) Freeze Early Layers
# =====================================================
for layer in generator.layers[:2]:
    layer.trainable = False

for layer in discriminator.layers[:2]:
    layer.trainable = False

# =====================================================
# 5. Loss Functions
# =====================================================
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

# =====================================================
# 6. Optimizers (LOW LR for Fine-Tuning)
# =====================================================
generator_optimizer = tf.keras.optimizers.Adam(1e-5, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-5, beta_1=0.5)

# =====================================================
# 7. Training Step (UNCHANGED)
# =====================================================
@tf.function
def train_step(images, labels):
    batch_size = tf.shape(images)[0]
    noise = tf.random.normal([batch_size, latent_dim])
    one_hot_labels = tf.one_hot(labels, num_classes)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(
            [noise, one_hot_labels], training=True
        )

        real_output = discriminator(
            [images, one_hot_labels], training=True
        )
        fake_output = discriminator(
            [generated_images, one_hot_labels], training=True
        )

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gen_grads = gen_tape.gradient(
        gen_loss, generator.trainable_variables
    )
    disc_grads = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables
    )

    generator_optimizer.apply_gradients(
        zip(gen_grads, generator.trainable_variables)
    )
    discriminator_optimizer.apply_gradients(
        zip(disc_grads, discriminator.trainable_variables)
    )

    return gen_loss, disc_loss

# =====================================================
# 8. Fine-Tuning Loop
# =====================================================
def fine_tune(dataset, epochs):
    for epoch in range(epochs):
        for image_batch, label_batch in dataset:
            g_loss, d_loss = train_step(
                image_batch, label_batch
            )

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"G Loss: {g_loss:.4f} | D Loss: {d_loss:.4f}"
        )

# =====================================================
# 9. Run Fine-Tuning
# =====================================================
fine_tune(dataset, epochs)

# =====================================================
# 10. Save Fine-Tuned Models
# =====================================================
generator.save("finetuned_generator.keras")
discriminator.save("finetuned_discriminator.keras")

print("✅ Fine-tuning complete & models saved")
