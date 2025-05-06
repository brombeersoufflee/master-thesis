import data_load
from sklearn.model_selection import GroupKFold
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv3D, BatchNormalization, GlobalAveragePooling3D, Dense
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
        print(f"{kfolds}-Fold split created")
        

    def build_model(self):
        if self.model_name == "CNN":
            input_shape = (64, 128, 64, 1)
            model = Sequential([
                Conv3D(64, kernel_size=7, strides=2, padding='same', activation='relu', input_shape=input_shape),
                BatchNormalization(),
                Conv3D(32, kernel_size=5, strides=1, padding='same', activation='relu'),
                BatchNormalization(),
                Conv3D(32, kernel_size=3, strides=1, padding='same', activation='relu'),
                BatchNormalization(),
                Conv3D(32, kernel_size=3, strides=1, padding='same', activation='relu'),
                BatchNormalization(),
                Conv3D(32, kernel_size=3, strides=1, padding='same', activation='relu'),
                BatchNormalization(),
                GlobalAveragePooling3D(),
                Dense(2, activation='softmax')
            ])

            print(model.summary())
            return model
        else:
            raise NotImplementedError(f"Model {self.model_name} is not implemented yet.")
        
    
    def train_model(self):
        if self.model_name == "CNN":
            model = self.build_model()
            model.compile(optimizer=Nadam(learning_rate=1e-4),  loss='categorical_crossentropy', metrics=[AUC(name='auc'),'accuracy'])
            # Add your training code here
            # model.fit(...)
            # TODO : check callback parameters
            callbacks = [
                EarlyStopping(monitor='val_auc', mode='max', patience=10, restore_best_weights=True),
                ModelCheckpoint('best_model.h5', monitor='val_auc', mode='max', save_best_only=True)
            ]

            # Data augmentation
            if self.augmentation:
                raise NotImplementedError("Data augmentation is not implemented yet.")
            else:
                pass

            historys = []
            for i, (train_idx, val_idx) in enumerate(self.cv_split):
                print(f"Fold {i+1}/{self.kfolds}")
                X_trains, X_vals = self.train_volumes[train_idx], self.train_volumes[val_idx]
                y_trains, y_vals = self.train_labels[train_idx], self.train_labels[val_idx]

                # making input data 5d
                X_trains = np.array([np.expand_dims(volume, axis=-1) for volume in X_trains])
                print("Xtrains shape", X_trains.shape)
                print("NAns", np.isnan(X_trains).sum())
                X_vals = np.array([np.expand_dims(volume, axis=-1) for volume in X_vals])
                print("Xvals shape", X_vals.shape)  

                # Train the model
                history = model.fit(
                    X_trains,
                    validation_data=y_trains,
                    epochs=100,
                    callbacks=callbacks,
                    verbose=1
                )


                # Predict on test data
                y_pred = model.predict(X_vals)
                auc_test = roc_auc_score(y_vals, y_pred)
                print(f"Test AUC: {auc_test:.4f}")
                # Save the model
                model.save(f"model_fold_{i+1}.h5")
                historys.append(history)
            return historys
        else:
            raise NotImplementedError(f"Model {self.model_name} is not implemented yet.")
