# This scipt will be used later on but the python noteboook is more convenient for development

import data_load
import numpy as np
import data_augmentation as aug

data_loader = data_load.DataLoader()
image_data, np_array_data, labels_data = data_loader.retina()
print("poaGlaucoma: ", sum(labels_data), "  Healthy:", len(labels_data)-sum(labels_data))
print("\n")

X_train, X_test, X_val, y_train, y_test, y_val = data_loader.retina_split(np_array_data,labels_data)
print("Train, Test, Validation size", len(X_train),len(X_test),len(X_val))
print("\n")

train_positive_pecentage = sum(y_train)/len(y_train)
test_positive_pecentage = sum(y_test)/len(y_test)
val_positive_pecentage = sum(y_val)/len(y_val)

print("train positives:", np.round(train_positive_pecentage,3), "  ---  test positives:", np.round(test_positive_pecentage,3), "  ---  val positives:", np.round(val_positive_pecentage,3))
print("\n")


glaucoma = sum(y_train)
glaucoma_cent = glaucoma/len(y_train)
healthy = len(y_train) - glaucoma
healthy_cent = healthy/len(y_train)
print("healthy: ", healthy," ---- healthy percentage:", np.round(healthy_cent, 4))
print("glaucoma: ", glaucoma," ---- glaucoma percentage:", np.round(glaucoma_cent, 4))
print("\n")


num_new_images = glaucoma-healthy
augmenter = aug.AugmentData(X_train, y_train)
print("Samples to produce in augmentation: ", num_new_images)
X_train_augmented, y_train_augmented = augmenter.augment_data(num_new_images, return_values="complete")


print("augmented data shape: ", X_train_augmented.shape, y_train_augmented.shape)
print("\n")
glaucoma = sum(y_train_augmented)
glaucoma_cent = glaucoma/len(y_train_augmented)
healthy = len(y_train_augmented) - glaucoma
healthy_cent = healthy/len(y_train_augmented)
print("aug healthy: ", healthy,"aug healthy percentage:", np.round(healthy_cent, 4))
print("aug glaucoma: ", glaucoma,"aug glaucoma percentage:", np.round(glaucoma_cent, 4))
print("\n")