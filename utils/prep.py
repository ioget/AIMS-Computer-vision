import os
import tensorflow as tf
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Intel Image Classification – 6 scene categories
CLASSES    = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
NUM_CLASSES = len(CLASSES)
IMG_SIZE   = 150
BATCH_SIZE = 32

# Paths are relative to the mnist/ working directory
TRAIN_DIR = os.path.join('data', 'archive', 'seg_train', 'seg_train')
TEST_DIR  = os.path.join('data', 'archive', 'seg_test',  'seg_test')

# ImageNet mean/std for normalization (images are natural scenes → good prior)
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


# ── PyTorch pipeline ──────────────────────────────────────────────────────────

def get_data_pytorch():
    """Return (train_loader, test_loader) for the Intel dataset."""

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        # ── Augmentation (train only) ──
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        # ── Normalization ──
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])

    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    test_dataset  = datasets.ImageFolder(TEST_DIR,  transform=test_transform)

    print(f"[PyTorch] Train samples : {len(train_dataset)}")
    print(f"[PyTorch] Test  samples : {len(test_dataset)}")
    print(f"[PyTorch] Classes       : {train_dataset.classes}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, test_loader


# ── TensorFlow pipeline ───────────────────────────────────────────────────────

def get_data_tensorflow():
    """Return (train_generator, test_generator) for the Intel dataset."""

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255,
        # ── Augmentation (train only) ──
        rotation_range=15,
        width_shift_range=0.05,
        height_shift_range=0.05,
        horizontal_flip=True,
        zoom_range=0.1,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest',
    )

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255,
    )

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASSES,
        shuffle=True,
    )

    test_gen = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASSES,
        shuffle=False,
    )

    return train_gen, test_gen
