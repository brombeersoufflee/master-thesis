import data_load
from keras_tuner import RandomSearch
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import StratifiedGroupKFold
import pandas as pd
import numpy as np
from keras.layers import Input
from keras.layers import Conv3D, BatchNormalization, GlobalAveragePooling3D, Dense, Activation, MaxPool3D, Dropout, Rescaling
from keras.optimizers import Nadam
from keras import Model
from keras.metrics import AUC


 # Data loading and preprocessing
data_loader = data_load.DataLoader()
print("Loading data...")
volumes, labels, patient_id, eye_side = data_loader.retina_npy(path ="datas/datas/")
print("Splitting data...")
train_volumes, test_volumes, train_labels, test_labels, train_patient_id, test_patient_id, train_eye_side, test_eye_side = data_loader.retina_npy_split(volumes, labels, patient_id, eye_side)
print(f"Train data shape: {train_volumes.shape}, Train labels shape: {train_labels.shape}")
print(f"Test data shape: {test_volumes.shape}, Test labels shape: {test_labels.shape}")
num_glaucoma = sum(train_labels)
num_healthy = len(train_labels) - num_glaucoma

print(train_volumes.nbytes)

gkf = StratifiedGroupKFold(n_splits=10, shuffle = True, random_state=53)
cv_splits = gkf.split(X=train_volumes, y=train_labels, groups=train_patient_id)
train_idx, val_idx = next(cv_splits)

# has to be after the split
df = pd.get_dummies(train_labels)
train_labels = df.values

X_trains, X_vals = train_volumes[train_idx], train_volumes[val_idx]
y_trains, y_vals = train_labels[train_idx], train_labels[val_idx]
train_patient_id = train_patient_id[train_idx]
val_patient_id = train_patient_id[val_idx]


# making input data 5d
X_trains = np.array([np.expand_dims(volume, axis=-1) for volume in X_trains])
print("Xtrains shape", X_trains.shape)
X_vals = np.array([np.expand_dims(volume, axis=-1) for volume in X_vals])
print("Xvals shape", X_vals.shape)  

# for i, volume in enumerate(X_trains):
#     X_trains[i] = np.expand_dims(volume, axis=-1)
# print("Xtrains shape", X_trains.shape)
# for i, volume in enumerate(X_vals):
#     X_vals[i] = np.expand_dims(volume, axis=-1)
# print("Xvals shape", X_vals.shape)  

# defining tuning model
def tuner_model(hp):
    return build_model(
        learning_rate=hp.Choice("learning_rate", [1e-4, 3e-4, 1e-3]),
        filters_1=hp.Choice("filters_1", [32, 64, 96]),
        filters_2=hp.Choice("filters_2", [32, 64]),
        kernel_size1=hp.Choice("kernel_size1", [5, 7]),
        kernel_size2=hp.Choice("kernel_size2", [3, 5]),
    )

tuner = RandomSearch(
    tuner_model,
    objective="val_auc",
    max_trials=20,
    executions_per_trial=1,
    directory="hyperband_logs",
    project_name="3d_cnn_tuning"
)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint("best_model.keras", save_best_only=True, monitor="val_loss", mode="min")
]

tuner.search(
    X_trains, y_trains,
    validation_data=(X_vals,y_vals),
    epochs=150,
    callbacks=callbacks
)

# Retrieve the best model
best_model = tuner.get_best_models(1)[0]


def build_model(self, lr = 1e-4,filters_1 = 64,filters_2 = 32, kernel_size1=7, kernel_size2 = 5):
    inputs = Input(shape=(64, 128, 64, 1))
    x = Conv3D(filters_1, kernel_size=kernel_size1, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv3D(filters_2, kernel_size=kernel_size2, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    for _ in range(3):
        x = Conv3D(filters_2, kernel_size=3, strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

    x = GlobalAveragePooling3D()(x)
    outputs = Dense(2, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    print(model.summary())
    model.compile(optimizer=Nadam(learning_rate=lr),  loss='categorical_crossentropy', metrics=[AUC(name='auc'),'f1_score', 'accuracy'])
    
    return model
