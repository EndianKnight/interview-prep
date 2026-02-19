> **TODO:** Expand this guide with detailed content.

# Deep Learning

Neural network architectures, training techniques, and the mathematical intuition behind deep learning.

## Topics to Cover

### Neural Network Fundamentals
- **Perceptron → MLP** — neurons, activation functions, forward pass, backpropagation
- **Activation functions** — ReLU, GELU, Sigmoid, Tanh, Swish — when to use each
- **Loss functions** — cross-entropy, MSE, triplet loss, contrastive loss
- **Backpropagation** — chain rule, computational graph, gradient flow
- **Vanishing/exploding gradients** — causes, solutions (ResNet skip connections, gradient clipping, batch norm)

### Core Architectures
- **CNNs** — convolutions, pooling, receptive field, ResNet, EfficientNet
- **RNNs/LSTMs/GRUs** — sequential data, gating mechanisms, limitations (long-range dependencies)
- **Autoencoders** — VAE (variational), denoising, latent space
- **GANs** — generator vs discriminator, training instability, mode collapse

### Training Techniques
- **Batch normalization** — internal covariate shift, layer norm vs batch norm vs group norm
- **Dropout** — regularization via random neuron deactivation, inference behavior
- **Learning rate scheduling** — warmup, cosine annealing, one-cycle policy
- **Mixed precision training** — FP16/BF16, loss scaling, memory savings
- **Data augmentation** — image transforms, text augmentation, Mixup, CutMix

### Transfer Learning
- **Pre-trained models** — ImageNet, language models as feature extractors
- **Fine-tuning strategies** — freeze layers, discriminative learning rates, gradual unfreezing
- **Domain adaptation** — when source and target distributions differ

### Distributed Training
- **Data parallelism** — replicate model, split batches
- **Model parallelism** — split model across GPUs
- **Pipeline parallelism** — split layers across GPUs with micro-batching
- **FSDP / DeepSpeed ZeRO** — shard optimizer state, gradients, parameters

### Interview Questions
- Explain backpropagation step by step
- Why do we need non-linear activation functions?
- BatchNorm vs LayerNorm — when to use each?
- How does ResNet solve the vanishing gradient problem?
- Explain transfer learning and when fine-tuning vs feature extraction
