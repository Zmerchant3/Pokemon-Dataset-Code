# Import Tensorflow 
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

# Load Pokemon dataset
**Next you load the entire dataset using the read_csv**

df = pd.read_csv('/kaggle/input/pokemon.csv')

# Display the first few rows of the dataset
print(df.head())

# Extract features (images) and labels
X = df['ImageURL']
y = df['Type1']

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Shuffle and split the data 
**Shuffle and split the data into training and testing sets**

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the image data
**You may need to adjust this based on your actual image data**

def preprocess_image(image_path):
    # Implement your image preprocessing logic here
    # Example: read the image, resize, normalize, etc.
    # Return the preprocessed image
    pass

X_train = X_train.apply(preprocess_image)
X_test = X_test.apply(preprocess_image)

# Convert the image data to numpy arrays
X_train = np.array(X_train.tolist())
X_test = np.array(X_test.tolist())

# Define your CNN model
**We define the CNN model using the model.add function**
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(height, width, channels)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluate the model
**Finally we evaluate the model**

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")
