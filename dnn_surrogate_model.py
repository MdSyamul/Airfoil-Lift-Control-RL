### Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from scipy import stats

### Importing the Dataset
dataset = pd.read_excel('Data for Airfoil.xlsx')

### Removing the Outliers
def remove_outliers_zscore(df, threshold=3):
    z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))  # Numeric columns only
    outliers = (z_scores > threshold).any(axis=1)
    print(f"Number of outliers detected: {outliers.sum()}")
    return df[~outliers]

dataset_clean = remove_outliers_zscore(dataset)

### Print max and min values of each variable after outlier removal
print("Maximum values of each variable after removing outliers:")
print(dataset_clean.max())
print("\nMinimum values of each variable after removing outliers:")
print(dataset_clean.min())

### Splitting Features (X) and Targets (y)
X = dataset_clean[['Mach', 'Alpha']].values  # Inputs
y = dataset_clean[['CL', 'CD']].values  # Targets

### Splitting into Training and Test Sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

### Feature Scaling: Standardization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

### Building the ANN model
ann = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=2, activation='linear')  # 2 outputs for cL and cD
])

### Compiling the ANN
ann.compile(optimizer='adam', 
            loss='mean_squared_error', 
            metrics=['mean_absolute_error', 'mean_absolute_percentage_error'])

### Early Stopping and Model Checkpoint
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, min_delta=1e-6, mode='min', verbose=1)
weights_of_min_loss = tf.keras.callbacks.ModelCheckpoint('best_weights.keras', monitor='loss', save_best_only=True, mode='min', verbose=1)

### Training the ANN
import time
start_time = time.time()
history = ann.fit(X_train, y_train, 
                  batch_size=32, 
                  epochs=1000, 
                  validation_data=(X_test, y_test), 
                  callbacks=[early_stopping, weights_of_min_loss], 
                  verbose=1)
end_time = time.time()
print(f"Training Time: {end_time - start_time:.5f} seconds")

### Plot Training and Validation Loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

### Predicting on Test Set
y_pred_test = ann.predict(X_test)
results_df = pd.DataFrame({
    'True CL': y_test[:, 0],
    'Predicted CL': y_pred_test[:, 0],
    'True CD': y_test[:, 1],
    'Predicted CD': y_pred_test[:, 1]
})
print(results_df)

### Save the Model
ann.save('cl_cd_prediction_ann.keras')
