import data_load
from sklearn.model_selection import GroupKFold
from tensorflow.keras import layers, models, Input
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

class Model_Implementation:
    def __init__(self, model_name, kfolds=10, augmentation=False):
        self.model_name = model_name
        self.kfolds = kfolds
        self.augmentation = augmentation

        # Data loading
        self.data_loader = data_load.DataLoader()
        self.np_arrays, self.labels_data, self.patient_id, self.eye_side = self.data_loader.retina_npy()
        gkf = GroupKFold(n_splits=self.kfolds, random_state=53)
        self.cv_split = gkf.split(X=self.np_arrays, y=self.labels_data, groups = self.patient_id)

    def build_model(self):
        if self.model_name == "CNN":
            inputs = Input(shape=self.np_arrays.shape)

            x = layers.Conv3D(32, kernel_size=7, strides=2, padding='same')(inputs)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)

            x = layers.Conv3D(32, kernel_size=5, strides=1, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)

            x = layers.Conv3D(32, kernel_size=3, strides=1, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)

            x = layers.Conv3D(32, kernel_size=3, strides=1, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)

            x = layers.Conv3D(32, kernel_size=3, strides=1, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)

            x = layers.GlobalAveragePooling3D()(x)
            outputs = layers.Dense(2, activation='softmax')(x)

            model = models.Model(inputs, outputs)
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

            for i, (train_idx, val_idx) in enumerate(self.cv_split):
                print(f"Fold {i+1}/{self.kfolds}")
                X_train, X_val = self.np_arrays[train_idx], self.np_arrays[val_idx]
                y_train, y_val = self.labels_data[train_idx], self.labels_data[val_idx]
    
                # Data augmentation
                if self.augmentation:
                    raise NotImplementedError("Data augmentation is not implemented yet.")
                else:
                    pass

                # Train the model
                history = model.fit(
                    X_train,
                    validation_data=y_train,
                    epochs=100,
                    callbacks=callbacks,
                    verbose=1
                )


                # Predict on test data
                y_pred = model.predict(X_val)
                auc_test = roc_auc_score(y_val, y_pred)
                print(f"Test AUC: {auc_test:.4f}")
        else:
            raise NotImplementedError(f"Model {self.model_name} is not implemented yet.")
