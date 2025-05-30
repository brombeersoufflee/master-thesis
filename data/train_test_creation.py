# DO NOT RUN IN THIS FOLDER, MUST BE RUN IN THE ROOT FOLDER
# This script was used to create a train-test split for the data, and just here for documentation purposes

# splits the data into train and test sets using GroupShuffleSplit
# The test size is set to 20% of the data, and the random state is set for reproducibility
# The data is split based on the patient_id to ensure that all OCT scans from a single patient are in either the train or test set, not both
# This is important for avoiding data leakage in medical imaging tasks
# inspired by https://stackoverflow.com/questions/54797508/how-to-generate-a-train-test-split-based-on-a-group-id

#import
from sklearn.model_selection import GroupShuffleSplit 
import numpy as np
import data_load

# loads original np_array data 
data_loader = data_load.DataLoader()
np_array_data, labels_data, patient_id, eye_side = data_loader.retina_npy()

# set parameters for the split
test_size = 0.2
n_splits = 1
random_state = 7

#splits the data into train and test sets using GroupShuffleSplit
splitter = GroupShuffleSplit(test_size=test_size, n_splits=n_splits, random_state = random_state)
split = splitter.split(np_array_data, groups=patient_id)
train_inds, test_inds = next(split)

# use the indices to create the train and test sets (only for the metrics below)
train = np_array_data[train_inds]
test = np_array_data[test_inds]

# if the split is sufficiently balanced, save the train and test indices to files and document the split and its parameters
if (sum(labels_data[train_inds])/len(train_inds) - sum(labels_data[test_inds])/len(test_inds)) < 0.05:
    print("The split is balanced")

    with open("data/train_test_split_documentation.txt", "a") as f:
        print("The split is balanced", file=f)
        print("\n", file=f)
        print(f"test_size: {test_size}, n_splits: {n_splits}, random_state: {random_state}", file=f)
        print("\n", file=f)
        print(f"train shape: {train.shape} - train %: {train.shape[0]/1110}  ---  test shape: {test.shape} test %: {test.shape[0]/1110}", file=f)
        print("\n", file=f)
        print("train positives:", sum(labels_data[train_inds])/len(train_inds), "  ---  test positives:", sum(labels_data[test_inds])/len(test_inds), file=f)

    np.savetxt("data/train_data_indeces.npy", train_inds)
    np.savetxt("data/test_data_indeces.npy", test_inds)
else:
    print("The split is not balanced")
    print("\n")
    print("rerun the code to get a balanced split")