import os
import SimpleITK as sitk
import json
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

class DataLoader:
    def __init__(self):
        self.path = "glaucoma_oct_data/"

    
    # load the data in folder glaucoma_oct_data/retina-oct-glaucoma
    # see data documentation on OneDrive for "Retina OCT Glaucoma dataset"
    def retina(self):
        """loads the data in folder glaucoma_oct_data/-oct-glaucoma
        see data documentation on OneDrive for "Retina OCT Glaucoma dataset"

        Returns
        -------
        images
            a list of 3D OCT images
        np_arrays
            a list of 3D OCT np arrays
        labels
            a list of labels corresponding via index to images and np_arrays
        """
        path = os.path.join(self.path,"retina-oct-glaucoma/retina-oct-glaucoma/imagesTr")
        json_path = os.path.join(self.path,"retina-oct-glaucoma/retina-oct-glaucoma/dataset.json")
        
        # Load JSON label info
        with open(json_path, "r") as f:
            metadata = json.load(f)
            label_data = metadata["training"]

        # Prepare outputs
        images = []
        np_arrays = []
        labels = []

        # Iterate through files and match with label
        for idx, filename in enumerate(os.listdir(path)):
            current_label_data = label_data[idx]
            current_label_image_name = int(current_label_data['image'][16:20])
            current_file_name = int(filename[5:9])
            if filename.endswith(".mha") and (current_label_image_name == current_file_name) and (current_file_name == idx):
                full_path = os.path.join(path, filename)
                image = sitk.ReadImage(full_path)
                images.append(image)
                np_arrays.append(sitk.GetArrayFromImage(image))
                labels.append(current_label_data["POAG"])
            else:
                print("Eithter non-mha object found:",filename, 
                      "OR indeces don't match ( current_label_image_name,current_file_name), idx",current_label_image_name,current_file_name,idx)
        
        return images, np_arrays, labels
    
    def retina_npy(self):
        """loads the data in folder glaucoma_oct_data/retina-oct-glaucoma-NPY
        see data documentation on OneDrive for "Retina OCT Glaucoma dataset"

        Returns
        -------
        np_arrays
            a list of 3D OCT np arrays
        pathology
            a list of labels corresponding via index to np_arrays
        patient_id
            a list of patient ids corresponding via index to np_arrays
        eye_side
            a list of eye sides (OD or OS) corresponding via index to np_arrays
        """

        path = os.path.join(self.path,"retina-oct-glaucoma-NPY")
        
        arrays = []
        pathology = []
        patient_id = []
        eye_side = []
    
        for filename in os.listdir(path):
            if filename.endswith('.npy'):
                file_path = os.path.join(path, filename)
                arr = np.load(file_path)
                arrays.append(arr)

                # Remove the .npy extension if needed
                name = filename[:-4] if filename.endswith('.npy') else filename
                parts = name.split('-')

                # Extract metadata
                pathology.append(parts[0])      # Normal or POAG
                patient_id.append(parts[1])      # The number after
                eye_side.append(parts[-1])       # OD or OS (last part)
        
        # Convert arrays to numpy array of objects
        np_arrays = np.array(arrays, dtype=int)
        lb = LabelBinarizer()
        labels_data = lb.fit_transform(pathology)
        if labels_data[0]!=0:
            print("Warning - labels_data[0] is not 0 -- inverted labels where POAG is 0!!!")

        return np_arrays, labels_data, patient_id, eye_side


    def retina_split(self, np_array_data, labels_data, test_proportion=0.2, val_proportion=0.15):
        """splits the dataset into parts for training, testing and validation

        Returns
        -------
        X_train, X_test, X_val
            a list of 3D OCT numpy arrays
        y_train, y_test, y_val
            a list of labels corresponding via index to the numpy arrays
        """
        X_train, X_test, y_train, y_test = train_test_split(np_array_data, labels_data, test_size=test_proportion, shuffle=True)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_proportion, shuffle=True)
        return X_train, X_test, X_val, y_train, y_test, y_val
    
    # subspended idea to save the data in json format
    # there might be different ways to do this, but I'm not sure it's necessary
    # def retina_save(self, X_train, X_test, X_val, y_train, y_test, y_val):
    #     train_json = [{"X": X_train[i], "y": y_train[i]} for i in range(len(X_train))]
    #     test_json = [{"X": X_test[i], "y": y_test[i]} for i in range(len(X_test))]
    #     val_json = [{"X": X_val[i], "y": y_val[i]} for i in range(len(X_val))]


    #     path = os.path.join(self.path,"retina-oct-glaucoma/retina-oct-glaucoma/")
    #     # Save the JSON files
        
    #     with open('train_data.json', 'w') as f:
    #         json.dump(train_json, f, indent=4)
    #     with open('test_data.json', 'w') as f:
    #         json.dump(test_json, f, indent=4)
    #     with open('val_data.json', 'w') as f:
    #         json.dump(val_json, f, indent=4)
    
    @staticmethod
    def retina_npy_split(X, y, groups, n_splits=10, shuffle=True, random_state=None):
        """splits the (retina_npy loaded) dataset into parts for training and testing
        X: np.ndarray
            The input data to be split.
        y: np.ndarray
            The target labels for the input data.
        groups: np.ndarray
            The group labels for the input data.
        n_splits: int, default=10
            The number of splits for cross-validation.
        shuffle: bool, default=True
            Whether to shuffle the data before splitting.
        random_state: int, default=None
            Random seed for reproducibility.
        Returns
        -------
        cv_split: generator
            A generator that yields train-test splits for cross-validation.
        """
        sgkf = GroupKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        cv_split = sgkf.split(X=X, y=y, groups = groups)
        return cv_split
    
    @staticmethod
    #https://www.geeksforgeeks.org/cross-validation-using-k-fold-with-scikit-learn/#logistic-regression-model-kfold-cross-validating
    def plot_kfold(cv_split, ax, data_size = 1110):
        """
        Plots the indices for a cross-validation object.

        Parameters:
        cv_split: Cross-validation split generator object
        data_size: Size of the dataset (default is 1110)
        ax: Matplotlib axis object
        """

        # Set color map for the plot
        cmap_cv = plt.cm.coolwarm
        n_splits = 0

        for i_split, (train_idx, test_idx) in enumerate(cv_split):
            # Create an array of NaNs and fill in training/testing indices
            indices = np.full(data_size, np.nan)
            indices[test_idx], indices[train_idx] = 1, 0
            
            # Plot the training and testing indices
            ax_x = range(len(indices))
            ax_y = [i_split + 0.5] * len(indices)
            ax.scatter(ax_x, ax_y, c=indices, marker="_", 
                        lw=10, cmap=cmap_cv, vmin=-0.2, vmax=1.2)
            n_splits += 1

        # Set y-ticks and labels
        y_ticks = np.arange(n_splits) + 0.5
        ax.set(yticks=y_ticks, yticklabels=range(n_splits),
                xlabel="X index", ylabel="Fold",
                ylim=[n_splits, -0.2], xlim=[0, data_size])

        # Set plot title and create legend
        ax.set_title("KFold", fontsize=14)
        legend_patches = [Patch(color=cmap_cv(0.8), label="Testing set"),
                            Patch(color=cmap_cv(0.02), label="Training set")]
        ax.legend(handles=legend_patches, loc=(1.03, 0.8))

    # load the data in folder glaucoma_oct_data/OCTandFundusImages
    # see data documentation on OneDrive for "Data on OCT and Fundus Images"
    def oct_fundus(self):
        raise NotImplementedError("This function is not implemented.")
    
    # load the data in folder glaucoma_oct_data/DatasetIEEE
    # see data documentation on OneDrive for "unnamed"
    def ieee_data(self):
        # relevant part is the OD (optic disc) folder for glaucoma data
        folder_structure = os.path.join(self.path,"DatasetIEEE/Dataset/Dataset/OD")
        raise NotImplementedError("This function is not implemented.")