import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import joblib

# Load improved features
X = np.load("X_features_improved.npy")
y = np.load("y_labels_improved.npy")

print(f"Improved data shape: X={X.shape}, y={y.shape}")

# Train model với features cải thiện
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model phức tạp hơn
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
              loss='binary_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(X_train, y_train, epochs=50, batch_size=32, 
                   validation_split=0.2, verbose=1)

# Save improved model
model.save("audio_classification_model_improved.keras")
joblib.dump(scaler, "scaler_improved.pkl")

# Evaluate
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Improved Model Test Accuracy: {test_accuracy:.4f}")