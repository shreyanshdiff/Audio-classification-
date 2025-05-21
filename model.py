import pandas as pd 
import numpy as np
import os
import librosa 
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf 
from tensorflow.keras import layers, models
import seaborn as sns 

np.random.seed(42)
tf.random.set_seed(42)

data_path = './dataset'

def extract_features(file_path, max_pad_len=174):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs = mfccs.T
    
        if mfccs.shape[0] < max_pad_len:
            pad_width = max_pad_len - mfccs.shape[0]
            mfccs = np.pad(mfccs, pad_width=((0, pad_width), (0, 0)), mode='constant')
        else:
            mfccs = mfccs[:max_pad_len, :]
        
        return mfccs
    except Exception as e:
        print(f"Error extracting features from {file_path}:{str(e)}")
        return None

def load_data(data_path):
    features = []
    labels = []
    file_paths = []
    
    print("Loading data from directory") 
    classes = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    
    if not classes:
        wav_files = [f for f in os.listdir(data_path) if f.endswith('.wav')]
        for wav_file in wav_files:
            file_path = os.path.join(data_path, wav_file)
            label = wav_file.split('_')[0]
            
            feature = extract_features(file_path)
            if feature is not None:
                features.append(feature)
                labels.append(label)
                file_paths.append(file_path)
    else:
        for class_label in classes:
            class_dir = os.path.join(data_path, class_label)
            wav_files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]
            
            for wav_file in wav_files:
                file_path = os.path.join(class_dir, wav_file)
                feature = extract_features(file_path)
                if feature is not None:
                    features.append(feature)
                    labels.append(class_label)
                    file_paths.append(file_path)
    
    if not features:
        raise ValueError("No valid audio files found in the dataset directory")
        
    features = np.array(features)
    labels = np.array(labels)
    print(f"Loaded {len(features)} samples from {len(set(labels))} classes.")
    unique_labels = set(labels)
    for label in unique_labels:
        count = np.sum(labels == label)
        print(f"Class '{label}': {count} samples")
    
    return features, labels, file_paths

def build_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def main():
    features, labels, file_paths = load_data(data_path)
    
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    features = features.reshape(features.shape[0], features.shape[1], features.shape[2], 1)
    
    x_train, x_test, y_train, y_test, train_paths, test_paths = train_test_split(
        features, encoded_labels, file_paths, test_size=0.2, random_state=42, stratify=encoded_labels
    )
    
    print(f"Training set: {x_train.shape[0]} samples")
    print(f"Testing set: {x_test.shape[0]} samples")
    
    train_df = pd.DataFrame({
        'file_path': train_paths,
        'label': label_encoder.inverse_transform(y_train)
    })
    test_df = pd.DataFrame({
        'file_path': test_paths,
        'label': label_encoder.inverse_transform(y_test)
    })
    
    train_df.to_csv('train.csv', index=False)
    test_df.to_csv("test.csv", index=False)

    input_shape = (x_train.shape[1], x_train.shape[2], 1)
    num_classes = len(label_encoder.classes_)
    model = build_model(input_shape, num_classes)
    
    print(model.summary())
    
    history = model.fit(
        x_train, y_train, 
        epochs=30, 
        batch_size=32, 
        validation_data=(x_test, y_test),
        callbacks=[
            EarlyStopping(
                patience=5, restore_best_weights=True
            )
        ]
    )    
    
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")
    
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    print("\nClassification Report")
    print(classification_report(y_test, y_pred_classes, 
                              target_names=label_encoder.classes_))  
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
              xticklabels=label_encoder.classes_,
              yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    
    model.save('audio_classification_model.h5')
    
    import pickle
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print("Model and encoder saved successfully!")
    
    return model, label_encoder

if __name__ == "__main__":
    main()
     
                
                
       
