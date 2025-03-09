# CycleGAN Project

This project implements a CycleGAN for unpaired image-to-image translation between Van Gogh-style paintings and real photos. It is optimized for training on a Windows machine with an NVIDIA RTX 3050 GPU (4GB VRAM) and includes a script for offline inference using trained models.

## Project Structure
cycle_gan/
├── dataset/                # Dataset folder (not included)
│   └── vangogh2photo/
│       ├── trainA/        # Van Gogh images (1231)
│       ├── trainB/        # Photo images (962)
│       ├── testA/         # Van Gogh test images (309)
│       ├── testB/         # Photo test images (238)
├── models/                # Trained model checkpoints
│   ├── G_vangogh2photo_B2A.pth.tar  # Photo -> Van Gogh
│   ├── G_vangogh2photo_A2B.pth.tar  # Van Gogh -> Photo
│   ├── critic_vangogh.pth.tar       # Discriminator for Van Gogh
│   └── critic_photo.pth.tar         # Discriminator for Photo
├── saved_images/          # Generated images during training
├── config.py             # Configuration settings
├── dataset.py            # Custom dataset loader
├── discriminator_model.py # Discriminator architecture
├── generator_model.py    # Generator architecture
├── train.py             # Training script
├── utils.py             # Utility functions
