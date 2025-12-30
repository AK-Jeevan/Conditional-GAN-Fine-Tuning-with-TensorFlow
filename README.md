# 🧠 Fine-Tuning a Pretrained Conditional GAN (CGAN)  
**TensorFlow / Keras**

This repository demonstrates how to **fine-tune a pretrained Conditional Generative Adversarial Network (CGAN)** using TensorFlow and Keras.

Instead of training a Conditional GAN from scratch, this project focuses on **GAN fine-tuning and domain adaptation**, showing how pretrained Generator and Discriminator models can be adapted to a new labeled dataset using low learning rates and stable adversarial training.

---

## 🚀 What This Project Covers

- Loading **pretrained CGAN Generator and Discriminator**
- True fine-tuning without modifying network architecture
- Class-conditional image generation using labels
- Image normalization to `[-1, 1]`
- Optional freezing of early layers for stability
- Custom adversarial training loop with `tf.GradientTape`
- Low learning-rate optimization for safe adaptation
- Saving fine-tuned models for downstream use

---

## 🧠 Why Fine-Tune a Pretrained CGAN?

This project helps you:

- Understand **Conditional GAN fine-tuning**
- Reuse learned visual representations
- Adapt generative models to new domains efficiently
- Improve training stability compared to training from scratch
- Apply **real-world GAN transfer learning workflows**

Fine-tuning is especially useful when:
- You already have a trained CGAN
- Training data is limited
- The new dataset is related but not identical to the original domain

---

## 🏗️ Training Architecture

### 🔹 Generator (Pretrained)
- Conditional image generator
- Input:
  - Random noise vector (`latent_dim = 128`)
  - One-hot encoded class labels
- Output: `64 × 64 × 3` RGB images
- Early layers can be frozen to preserve learned features

### 🔹 Discriminator (Pretrained)
- Conditional binary classifier (real vs fake)
- Input:
  - Image (`64 × 64 × 3`)
  - Corresponding class label
- Trained jointly with the generator during fine-tuning

---

## 🧪 Fine-Tuning Strategy

- Architecture remains **unchanged**
- Low learning rates prevent catastrophic forgetting
- Early layers can be frozen for stability
- Labels are injected into both Generator and Discriminator
- Adversarial loss is optimized using separate gradient tapes

---

## 📉 Loss Functions

**Binary Cross-Entropy (from logits)**

### Generator Loss
- Encourages generated images to be classified as **real**

### Discriminator Loss
- Real images → label **1**
- Generated images → label **0**

### Optimizers
- Adam optimizer for both networks
- Learning rate: `1e-5`
- `beta_1 = 0.5` for stable GAN training

---

## 🔍 Key Concepts Demonstrated

- Conditional GAN fine-tuning
- Label-aware generative modeling
- GAN transfer learning
- Custom training loops in TensorFlow
- Training stability techniques
- Layer freezing strategies

---

## 💾 Output Artifacts

After fine-tuning, the script saves:

- `finetuned_generator.keras`
- `finetuned_discriminator.keras`

These models can be reused for **conditional image generation** or further fine-tuning.

---

## ⚠️ Important Notes

- This project **fine-tunes** a pretrained CGAN — it does not train one from scratch
- Pretrained models must match the class count and architecture
- Low learning rates are critical for preserving learned features
- Label consistency between dataset and models is required

---

## 📜 License

MIT License

---

## ⭐ Support

If this repository helped you:

⭐ Star the repo  
🧠 Share it with other GAN and deep learning learners  
🚀 Use it as a foundation for advanced conditional generation projects
