# utils.py
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

def get_image_datasets(data_dir, img_size=(224,224), batch_size=32, val_split=0.2, seed=123, subset_train=True):
    """
    If data_dir contains 'train' and 'val' subfolders, this function will load them.
    Otherwise it will use image_dataset_from_directory with validation_split.
    """
    import os
    if os.path.isdir(f"{data_dir}/train") and os.path.isdir(f"{data_dir}/val"):
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            f"{data_dir}/train",
            image_size=img_size,
            batch_size=batch_size
        )
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            f"{data_dir}/val",
            image_size=img_size,
            batch_size=batch_size
        )
    else:
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split=val_split,
            subset="training",
            seed=seed,
            image_size=img_size,
            batch_size=batch_size
        )
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split=val_split,
            subset="validation",
            seed=seed,
            image_size=img_size,
            batch_size=batch_size
        )

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds

def plot_history(history, title='Training'):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='train loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='val loss')
    plt.legend()
    plt.title(title + ' Loss')

    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'], label='train acc')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='val acc')
    plt.legend()
    plt.title(title + ' Accuracy')
    plt.show()
