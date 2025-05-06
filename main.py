# This scipt will be used later on but the python noteboook is more convenient for development

import data_load
import numpy as np
import data_augmentation as aug
import model_implementation as model_impl
from sklearn.model_selection import GroupKFold


data_loader = data_load.DataLoader()
# image_data, np_array_data, labels_data = data_loader.retina()
np_array_data, labels_data, patient_id, eye_side = data_loader.retina_npy()
X_train, X_test, y_train, y_test, train_patient_id, test_patient_id, train_eye_side, test_eye_side = data_loader.retina_npy_split(np_array_data, labels_data, patient_id, eye_side)


# print("poaGlaucoma: ", sum(labels_data), "  Healthy:", len(labels_data)-sum(labels_data))
# print("\n")

# train_positive_pecentage = sum(y_train)/len(y_train)
# test_positive_pecentage = sum(y_test)/len(y_test)

# print("train positives:", np.round(train_positive_pecentage,3), "  ---  test positives:", np.round(test_positive_pecentage,3))
# print("\n")


# glaucoma = sum(y_train)
# glaucoma_cent = glaucoma/len(y_train)
# healthy = len(y_train) - glaucoma
# healthy_cent = healthy/len(y_train)
# print("healthy: ", healthy," ---- healthy percentage:", np.round(healthy_cent, 4))
# print("glaucoma: ", glaucoma," ---- glaucoma percentage:", np.round(glaucoma_cent, 4))
# print("\n")


# num_new_images = glaucoma-healthy
# augmenter = aug.AugmentData(X_train, y_train)
# print("Samples to produce in augmentation: ", num_new_images)
# X_train_augmented, y_train_augmented = augmenter.augment_data(num_new_images, return_values="complete")


# print("augmented data shape: X", X_train_augmented.shape,"----- y", y_train_augmented.shape)
# print("\n")
# glaucoma = sum(y_train_augmented)
# glaucoma_cent = glaucoma/len(y_train_augmented)
# healthy = len(y_train_augmented) - glaucoma
# healthy_cent = healthy/len(y_train_augmented)
# print("aug healthy: ", healthy,"aug healthy percentage:", np.round(healthy_cent, 4))
# print("aug glaucoma: ", glaucoma,"aug glaucoma percentage:", np.round(glaucoma_cent, 4))
# print("\n")


print("Xtrain",X_train.shape)

kfolds = 10
X_train_split = X_train.copy()
y_train_split = y_train.copy()

gkf = GroupKFold(n_splits=kfolds, shuffle= True, random_state=53)
cv_split = gkf.split(X=X_train_split, y=y_train_split, groups=train_patient_id)
print(f"{kfolds}-Fold split created")

for i, (train_idx, val_idx) in enumerate(cv_split):
    
    print("Xtrain",X_train.shape)
    print("XTRAIN 795", X_train[795].shape)
    X_train_split, X_val = X_train[train_idx], X_train[val_idx]
    y_train_split, y_val = y_train[train_idx], y_train[val_idx]

    print(X_train_split.shape[1:])
    print(X_val.shape[1:])
    print(y_train_split.shape)
    print(y_val.shape)

    print(f"Fold {i+1}/{kfolds}")