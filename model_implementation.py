import data_load
from sklearn.preprocessing import  OneHotEncoder
import pandas as pd
from sklearn.model_selection import GroupKFold
import tensorflow as tf
from keras import Model
from keras.utils import plot_model
from keras.layers import Input
from keras.models import Sequential
from keras.layers import Conv3D, BatchNormalization, GlobalAveragePooling3D, Dense, Activation
from keras.optimizers import Nadam
from keras.metrics import AUC
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from sklearn.metrics import roc_auc_score

class Model_Implementation:
    def __init__(self, model_name, kfolds=10, augmentation=False):
        self.model_name = model_name
        self.kfolds = kfolds
        self.augmentation = augmentation

        # Data loading
        self.data_loader = data_load.DataLoader()
        print("Loading data...")
        self.volumes, self.labels, self.patient_id, self.eye_side = self.data_loader.retina_npy()
        print("Splitting data")
        self.train_volumes, self.test_volumes, self.train_labels, self.test_labels, self.train_patient_id, self.test_patient_id, self.train_eye_side, self.test_eye_side = self.data_loader.retina_npy_split(self.volumes, self.labels, self.patient_id, self.eye_side)
        print(f"Train data shape: {self.train_volumes.shape}, Train labels shape: {self.train_labels.shape}")
        print(f"Test data shape: {self.test_volumes.shape}, Test labels shape: {self.test_labels.shape}")

        gkf = GroupKFold(n_splits=self.kfolds, shuffle= True, random_state=53)
        self.cv_split = gkf.split(X=self.train_volumes, y=self.train_labels, groups=self.train_patient_id)
        # labels = np.array(self.train_labels).reshape(-1, 1)  # Must be 2D
        # encoder = OneHotEncoder()
        # self.train_labels = encoder.fit_transform(labels)
        # print(self.train_labels)
        df = pd.get_dummies(self.train_labels)
        self.train_labels = df.values
        print(self.train_labels)
        print(f"{kfolds}-Fold split created")
        

    def build_model(self):
        if self.model_name == "CNN":
            inputs = Input(shape=(64, 128, 64, 1))
            x = Conv3D(64, kernel_size=7, strides=2, padding='same')(inputs)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
            x = Conv3D(32, kernel_size=5, strides=1, padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
            x = Conv3D(32, kernel_size=3, strides=1, padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
            x = Conv3D(32, kernel_size=3, strides=1, padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
            x = Conv3D(32, kernel_size=3, strides=1, padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
            x = GlobalAveragePooling3D()(x)
            outputs = Dense(2, activation='softmax')(x)
            
            return Model(inputs, outputs)
        else:
            raise NotImplementedError(f"Model {self.model_name} is not implemented yet.")
        
    
    def train_model(self):
        if self.model_name == "CNN":
           
            # Data augmentation
            if self.augmentation:
                raise NotImplementedError("Data augmentation is not implemented yet.")
            else:
                pass

            historys = []
            for i, (train_idx, val_idx) in enumerate(self.cv_split):
                model = self.build_model()

                print(model.summary())
                plot_model(model, show_shapes=True)
                model.compile(optimizer=Nadam(learning_rate=1e-4),  loss='categorical_crossentropy', metrics=[AUC(name='auc'),'accuracy'])
                # Add your training code here
                # model.fit(...)
                # TODO : check callback parameters
                callbacks = [
                    EarlyStopping(monitor='val_auc', mode='max', patience=10, restore_best_weights=True),
                    ModelCheckpoint('best_model.keras', monitor='val_auc', mode='max', save_best_only=True)
                ]

                print(f"Fold {i+1}/{self.kfolds}")
                X_trains, X_vals = self.train_volumes[train_idx], self.train_volumes[val_idx]
                y_trains, y_vals = self.train_labels[train_idx], self.train_labels[val_idx]

                # making input data 5d
                X_trains = np.array([np.expand_dims(volume, axis=-1) for volume in X_trains])
                print("Xtrains shape", X_trains.shape)
                X_vals = np.array([np.expand_dims(volume, axis=-1) for volume in X_vals])
                print("Xvals shape", X_vals.shape)  

                X_y_trains = tf.data.Dataset.from_tensor_slices(({'input_values': X_trains},y_trains))
                X_y_vals = tf.data.Dataset.from_tensor_slices(({'input_values': X_vals},y_vals))

                # Train the model
                history = model.fit(x = X_trains,
                    y = y_trains,
                    validation_data=(X_vals, y_vals),
                    epochs=100,
                    callbacks=callbacks,
                    verbose=1
                    )


                # Predict on test data
                y_pred = model.predict(X_vals)
                auc_test = roc_auc_score(y_vals, y_pred)
                print(f"Test AUC: {auc_test:.4f}")
                # Save the model
                #  WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. 
                # We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`.
                model.save(f"model_fold_{i+1}.keras")
                historys.append(history)
                tf.keras.backend.clear_session()
            return historys
        else:
            raise NotImplementedError(f"Model {self.model_name} is not implemented yet.")
