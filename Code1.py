import os
import pydicom
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns

# Dosya yolu
dataset_path = 'C:/Users/talha/Desktop/DS1'
metadata_path = os.path.join(dataset_path, 'veribilgisi.xlsx')

# Veri dosyasını yükle
metadata = pd.read_excel(metadata_path)

def load_and_preprocess_dcm(file_path, label_name):
    dcm_data = pydicom.dcmread(file_path)
    img = dcm_data.pixel_array
    
    # Normalizasyon
    img_normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    
    # Boyutlandırma
    img_resized = cv2.resize(img_normalized, (256, 256))
    
    # Gaussian Blur
    img_blurred = cv2.GaussianBlur(img_resized, (5, 5), 0)
    
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_blurred.astype(np.uint8))
    
    # Kontrast ve Parlaklık Ayarlama
    alpha = 1.5  # Kontrast kontrolü (1.0-3.0 arası değerler)
    beta = 20    # Parlaklık kontrolü (0-100 arası değerler)
    img_adjusted = cv2.convertScaleAbs(img_clahe, alpha=alpha, beta=beta)
    
    # Kenar belirleme (Edge Detection)
    img_edges = cv2.Canny(img_adjusted, threshold1=30, threshold2=100)
    
    # Sobel filtresi ile kenar belirleme
    sobelx = cv2.Sobel(img_adjusted, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img_adjusted, cv2.CV_64F, 0, 1, ksize=5)
    img_sobel = cv2.sqrt(cv2.addWeighted(np.float32(cv2.convertScaleAbs(sobelx)), 0.5, np.float32(cv2.convertScaleAbs(sobely)), 0.5, 0))
    
    # Gereksiz kısımları kırpma
    non_zero_pixels = cv2.findNonZero(img_sobel)
    if non_zero_pixels is not None:
        x, y, w, h = cv2.boundingRect(non_zero_pixels)
        img_cropped = img_adjusted[y:y+h, x:x+w]
    else:
        img_cropped = img_adjusted
    
    img_final = cv2.resize(img_cropped, (256, 256))
    
    # Görüntüyü ekranda göster
    plt.imshow(img_final, cmap='gray')
    plt.title(f'Pre-processed Image: {label_name}')
    plt.axis('off')
    plt.show(block=False)
    plt.pause(0.001)
    plt.close()

    return img_final / 255.0  # Normalize edilmiş görüntü

def load_dataset(metadata, dataset_path):
    images = []
    birads_labels = []
    lesion_labels = []

    for _, row in metadata.iterrows():
        category = row['Category']
        patient_id = str(row['Patient_id'])
        file_name = row['File_name']
        label_name = row['Label_name']
        
        file_path = os.path.join(dataset_path, category, patient_id, file_name)
        
        if os.path.exists(file_path):
            img = load_and_preprocess_dcm(file_path, label_name)
            images.append(img)
            birads_labels.append(category)
            lesion_labels.append(label_name)
    
    return np.array(images), np.array(birads_labels), np.array(lesion_labels)

images, birads_labels, lesion_labels = load_dataset(metadata, dataset_path)
images = np.expand_dims(images, axis=-1)  # Kanal boyutunu ekle

# Etiketleri kodlama
birads_encoder = LabelEncoder()
birads_labels_encoded = birads_encoder.fit_transform(birads_labels)
birads_labels_categorical = to_categorical(birads_labels_encoded)

lesion_encoder = LabelEncoder()
lesion_labels_encoded = lesion_encoder.fit_transform(lesion_labels)
lesion_labels_categorical = to_categorical(lesion_labels_encoded)

# Eğitim ve test verilerini ayırma
X_train, X_test, y_birads_train, y_birads_test, y_lesion_train, y_lesion_test = train_test_split(
    images, birads_labels_categorical, lesion_labels_categorical, test_size=0.2, random_state=42)

# Model tanımlama
def create_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(4, activation='softmax', name='birads_output'),  # BIRADS için 4 çıktı
        Dense(3, activation='softmax', name='lesion_output')   # Lezyon tipi için 3 çıktı
    ])
    return model

input_shape = (256, 256, 1)
model = create_model(input_shape)

# Model derleme
model.compile(optimizer='adam', 
              loss={'birads_output': 'categorical_crossentropy', 'lesion_output': 'categorical_crossentropy'}, 
              metrics={'birads_output': 'accuracy', 'lesion_output': 'accuracy'})

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Model eğitme
history = model.fit(X_train, {'birads_output': y_birads_train, 'lesion_output': y_lesion_train}, 
                    epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Model değerlendirme
model.evaluate(X_test, {'birads_output': y_birads_test, 'lesion_output': y_lesion_test})

# Tahmin yapma
y_birads_pred = model.predict(X_test)[0]
y_lesion_pred = model.predict(X_test)[1]

# Sınıflandırma raporu
print(classification_report(y_birads_test.argmax(axis=1), y_birads_pred.argmax(axis=1), target_names=birads_encoder.classes_))
print(classification_report(y_lesion_test.argmax(axis=1), y_lesion_pred.argmax(axis=1), target_names=lesion_encoder.classes_))

# Confusion Matrix
birads_cm = confusion_matrix(y_birads_test.argmax(axis=1), y_birads_pred.argmax(axis=1))
lesion_cm = confusion_matrix(y_lesion_test.argmax(axis=1), y_lesion_pred.argmax(axis=1))

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
sns.heatmap(birads_cm, annot=True, fmt='d', cmap='Blues', xticklabels=birads_encoder.classes_, yticklabels=birads_encoder.classes_)
plt.title('BIRADS Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.subplot(1, 2, 2)
sns.heatmap(lesion_cm, annot=True, fmt='d', cmap='Blues', xticklabels=lesion_encoder.classes_, yticklabels=lesion_encoder.classes_)
plt.title('Lesion Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.show()

# ROC Curve ve AUC
def plot_roc_curve(y_test, y_pred, n_classes, title):
    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (area = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title(title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.show()

plot_roc_curve(y_birads_test, y_birads_pred, 4, 'BIRADS ROC Curve')
plot_roc_curve(y_lesion_test, y_lesion_pred, 3, 'Lesion ROC Curve')

# Eğitim geçmişi grafikleri
plt.figure(figsize=(14, 5))

plt.subplot(1, 3, 1)
plt.plot(history.history['birads_output_accuracy'], label='BIRADS Train Accuracy')
plt.plot(history.history['val_birads_output_accuracy'], label='BIRADS Val Accuracy')
plt.plot(history.history['lesion_output_accuracy'], label='Lesion Train Accuracy')
plt.plot(history.history['val_lesion_output_accuracy'], label='Lesion Val Accuracy')
plt.title('Accuracy per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(history.history['birads_output_loss'], label='BIRADS Train Loss')
plt.plot(history.history['val_birads_output_loss'], label='BIRADS Val Loss')
plt.plot(history.history['lesion_output_loss'], label='Lesion Train Loss')
plt.plot(history.history['val_lesion_output_loss'], label='Lesion Val Loss')
plt.title('BIRADS & Lesion Loss per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
