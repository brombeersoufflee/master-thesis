import data_load
from tensorflow.keras import layers, models, Input
from sklearn.model_selection import GroupKFold
from tensorflow.keras.optimizers import Nadam

class Model_Implementation:
    def __init__(self, model_name, kfolds=10):
        self.model_name = model_name
        self.kfolds = kfolds

        # Data loading
        self.data_loader = data_load.DataLoader()
        self.np_arrays, self.labels_data, self.patient_id, self.eye_side = self.data_loader.retina_npy()

    
    def preprocess_data(self):
        if self.model_name == "CNN":
            gkf = GroupKFold(n_splits=self.kfolds, random_state=53)
            cv_split = gkf.split(X=self.np_arrays, y=self.labels_data, groups = self.patient_id)

    

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
            model.compile(optimizer=Nadam(learning_rate=1e-4),  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            # Add your training code here
            # model.fit(...)
        else:
            raise NotImplementedError(f"Model {self.model_name} is not implemented yet.")
