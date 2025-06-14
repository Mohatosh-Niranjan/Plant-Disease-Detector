import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.efficientnet import preprocess_input, EfficientNetB0
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def create_model(img_size, num_classes):
    """
    Create a more robust model architecture.
    """
    # Use EfficientNetB0 as base model
    base_model = EfficientNetB0(
        input_shape=(*img_size, 3),
        include_top=False,
        weights='imagenet'
        
    )

    # Freeze base layers for feature extraction
    base_model.trainable = False

    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    return Model(inputs=base_model.input, outputs=predictions)

def train_model(data_dir, img_size=(224, 224), batch_size=32):
    """
    Train the plant disease detection model with improved setup.
    """

    # Use EfficientNet preprocessing
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.2,
        rotation_range=30,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2
    )

    # Load training data
    train_data = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        subset='training',
        class_mode='categorical',
        shuffle=True
    )

    val_data = val_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        subset='validation',
        class_mode='categorical',
        shuffle=False
    )

    print("Class Indices:", train_data.class_indices)

    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(train_data.classes),
                                         y=train_data.classes)
    class_weights_dict = dict(enumerate(class_weights))

    # Create and compile model
    model = create_model(img_size, len(train_data.class_indices))
    
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Enhanced callbacks
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1),
        ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
    ]

    # Initial Training Phase
    print("Starting initial training phase...")
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=30,
        callbacks=callbacks,
        class_weight=class_weights_dict,
        verbose=1
    )

    # Fine-tuning Phase
    print("\nStarting fine-tuning phase...")
    for layer in model.layers[-20:]:
        layer.trainable = True

    model.compile(optimizer=Adam(learning_rate=1e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history_fine = model.fit(
        train_data,
        validation_data=val_data,
        epochs=20,
        callbacks=callbacks,
        class_weight=class_weights_dict,
        verbose=1
    )

    return model, history, history_fine, val_data

def evaluate_model(model, val_data):
    """
    Evaluate model performance on validation set.
    """
    val_loss, val_acc = model.evaluate(val_data)
    print(f'\nOverall Validation Accuracy: {val_acc * 100:.2f}%')

    y_true = val_data.classes
    y_pred = np.argmax(model.predict(val_data), axis=1)

    class_names = list(val_data.class_indices.keys())

    cm = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred, target_names=class_names, digits=4)

    print('\nDetailed Classification Report:')
    print(class_report)

    print('\nPer-class Accuracy:')
    per_class_accuracies = cm.diagonal() / cm.sum(axis=1)
    for name, acc in zip(class_names, per_class_accuracies):
        print(f"{name}: {acc * 100:.2f}%")

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

if __name__ == "__main__":
    # Update path based on your environment
    # data_dir = "/kaggle/input/plant-village/PlantVillage"  # For Kaggle
    data_dir = "C:/Users/Acer/Downloads/PlantVillage"  # For local use

    print("Starting model training...")
    model, history, history_fine, val_data = train_model(data_dir)

    print("\nSaving final model...")
    model.save('plant_disease_efficientnetb0')
    print("Model saved as 'plant_disease_efficientnetb0' (SavedModel format)")

    print("\nEvaluating model performance...")
    evaluate_model(model, val_data)

    print("\nPlotting training history...")
    def plot_training_history(history, history_fine):
        acc = history.history['accuracy'] + history_fine.history['accuracy']
        val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
        loss = history.history['loss'] + history_fine.history['loss']
        val_loss = history.history['val_loss'] + history_fine.history['val_loss']

        epochs_range = range(1, len(acc) + 1)

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.tight_layout()
        plt.show()

    plot_training_history(history, history_fine)