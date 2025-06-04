import data_load
import data_augmentation
from sklearn.preprocessing import  OneHotEncoder
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import tensorflow as tf
from keras import Model
from keras.utils import plot_model
from keras.layers import Input
from keras.models import Sequential
from keras.layers import Conv3D, BatchNormalization, GlobalAveragePooling3D, Dense, Activation, MaxPool3D, Dropout
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

        # Data loading and preprocessing
        self.data_loader = data_load.DataLoader()
        print("Loading data...")
        self.volumes, self.labels, self.patient_id, self.eye_side = self.data_loader.retina_npy()
        print("Splitting data...")
        self.train_volumes, self.test_volumes, self.train_labels, self.test_labels, self.train_patient_id, self.test_patient_id, self.train_eye_side, self.test_eye_side = self.data_loader.retina_npy_split(self.volumes, self.labels, self.patient_id, self.eye_side)
        print(f"Train data shape: {self.train_volumes.shape}, Train labels shape: {self.train_labels.shape}")
        print(f"Test data shape: {self.test_volumes.shape}, Test labels shape: {self.test_labels.shape}")
        self.num_glaucoma = sum(self.train_labels)
        self.num_healthy = len(self.train_labels) - self.num_glaucoma


        print(self.train_volumes.nbytes)

        gkf = StratifiedGroupKFold(n_splits=self.kfolds, shuffle= True, random_state=53)
        self.cv_split = gkf.split(X=self.train_volumes, y=self.train_labels, groups=self.train_patient_id)
        print(f"{kfolds}-Fold split created")

        # has to be after the split
        df = pd.get_dummies(self.train_labels)
        self.train_labels = df.values

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
        
        elif self.model_name == "CNNAlex":
            inputs = Input(shape=(64, 128, 64, 1))
            x = Conv3D(64, kernel_size=7, strides=2, padding='same', activation = 'relu')(inputs)
            x = MaxPool3D(pool_size=2)(x)
            x = BatchNormalization()(x)
            x = Conv3D(64, kernel_size=5, strides=1, padding='same', activation="relu")(x)
            x = MaxPool3D(pool_size=2)(x)
            x = BatchNormalization()(x)
            x = Conv3D(64, kernel_size=3, strides=1, padding='same', activation="relu")(x)
            x = MaxPool3D(pool_size=2)(x)
            x = BatchNormalization()(x)
            x = Conv3D(64, kernel_size=3, strides=1, padding='same', activation="relu")(x)
            x = MaxPool3D(pool_size=2)(x)
            x = BatchNormalization()(x)
            x = Conv3D(64, kernel_size=3, strides=1, padding='same', activation="relu")(x)
            x = MaxPool3D(pool_size=2)(x)
            x = BatchNormalization()(x)
            x = GlobalAveragePooling3D()(x)
            x = Dense(units=512, activation="relu")(x)
            x = Dropout(0.3)(x)
            x = Dense(units=256, activation="relu")(x)
            x = Dropout(0.3)(x)
            x = Dense(units=128, activation="relu")(x) 
            x = Dropout(0.3)(x)
            outputs = Dense(2, activation='softmax')(x)
            return Model(inputs, outputs)
        else:
            raise NotImplementedError(f"Model {self.model_name} is not implemented yet.")
        
    
    def train_model(self):                   
        historys = []
        print(self.train_labels.shape)
        for i, (train_idx, val_idx) in enumerate(self.cv_split):
            
            np.savetxt(f"{self.model_name}_split{i}_train_data_indeces.npy", train_idx)
            np.savetxt(f"{self.model_name}_split{i}_val_data_indeces.npy", val_idx) 
            
            model = self.build_model()

            print(model.summary())
            plot_model(model, show_shapes=True)
            model.compile(optimizer=Nadam(learning_rate=1e-4),  loss='categorical_crossentropy', metrics=[AUC(name='auc'),'accuracy'])
            # Add your training code here
            # model.fit(...)
            # TODO : check callback parameters
            callbacks = [
                EarlyStopping(monitor='val_loss', mode='min', patience=7, restore_best_weights=True),
                ModelCheckpoint('best_model.keras', monitor='val_loss', mode='min', save_best_only=True)
            ]

            print(f"Fold {i+1}/{self.kfolds}")
            X_trains, X_vals = self.train_volumes[train_idx], self.train_volumes[val_idx]
            y_trains, y_vals = self.train_labels[train_idx], self.train_labels[val_idx]
            train_patient_id = self.train_patient_id[train_idx]
            val_patient_id = self.train_patient_id[val_idx]

            print("xtrain b4 5d shape", X_trains.nbytes)
            # Data augmentation
            if self.augmentation:
                print("Data augmentation is enabled.")
                num_new_volumes = self.num_glaucoma-self.num_healthy
                augmenter = data_augmentation.AugmentData(X_trains, y_trains, train_patient_id)
                print(f"Original train data shape: {self.train_volumes.shape}, Original train labels shape: {self.train_labels.shape}")
                X_trains, y_trains, train_patient_id = augmenter.augment_data(num_new_volumes, return_values= "complete")
                print("Augmentation completed.")
                print(f"Augmented train data shape: {X_trains.shape}, Augmented train labels shape: {y_trains.shape}")
            else:
                print("Data augmentation is disabled.")
                print(f"X train data shape: {X_trains.shape}, y train data shape: {y_trains.shape}")
            print(f"X validation shape: {X_vals.shape}, y validation shape: {y_vals.shape}")
            
            # train_patient_id = None # Google colab RAM

            print(f"Train data labels positive:{np.where(y_trains[:, 1] == False)[0].shape[0]} / {y_trains.shape[0]}")

            # making input data 5d
            X_trains = np.array([np.expand_dims(volume, axis=-1) for volume in X_trains])
            print("Xtrains shape", X_trains.shape)
            X_vals = np.array([np.expand_dims(volume, axis=-1) for volume in X_vals])
            print("Xvals shape", X_vals.shape)  

            for i, volume in enumerate(X_trains):
                X_trains[i] = np.expand_dims(volume, axis=-1)
            print("Xtrains shape", X_trains.shape)
            for i, volume in enumerate(X_vals):
                X_vals[i] = np.expand_dims(volume, axis=-1)
            print("Xvals shape", X_vals.shape)  



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
 