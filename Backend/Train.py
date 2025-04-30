!pip install tensorflow opencv-python-headless matplotlib pandas lxml Cython pycocotools
!git clone https://github.com/tensorflow/models.git
%cd models/research
!protoc object_detection/protos/*.proto --python_out=.
!cp object_detection/packages/tf2/setup.py .
!pip install .

# Step 1: Import Required Libraries
import os
import zipfile
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Step 2: Unzip Dataset
zip_path = '/content/sample_data/Dataset.zip'
extract_path = '/content/sample_data/'

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

dataset_dir = '/content/sample_data/Dataset'
train_dir = os.path.join(dataset_dir, 'train')
test_dir = os.path.join(dataset_dir, 'test', 'valid')

# Step 3: Load Annotations
train_annotations_path = os.path.join(train_dir, '_annotations.csv')
test_annotations_path = os.path.join(test_dir, '_annotations.csv')

train_annotations = pd.read_csv(train_annotations_path)
test_annotations = pd.read_csv(test_annotations_path)

# Step 4: Prepare Data for Training
if 'class' in train_annotations.columns:
    train_annotations.rename(columns={'class': 'label'}, inplace=True)
if 'class' in test_annotations.columns:
    test_annotations.rename(columns={'class': 'label'}, inplace=True)

# Append full paths to filenames
train_annotations['filename'] = train_annotations['filename'].apply(lambda x: os.path.join(train_dir, x))
test_annotations['filename'] = test_annotations['filename'].apply(lambda x: os.path.join(test_dir, x))

# Split training data into train and validation sets
train_df, val_df = train_test_split(train_annotations, test_size=0.2, random_state=42, stratify=train_annotations['label'])

# Step 5: Data Generators
img_height, img_width = 150, 150
batch_size = 32

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='filename',
    y_col='label',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col='filename',
    y_col='label',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# Step 6: Build the Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(train_generator.class_indices), activation='softmax')
])

# Step 7: Compile the Model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Step 8: Train the Model
epochs = 10
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator
)

# Step 9: Save the Trained Model
model.save('/content/trained_model.h5')

# Step 10: Download the Model
from google.colab import files
files.download('/content/trained_model.h5')
