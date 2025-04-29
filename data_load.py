import os
import SimpleITK as sitk
import json
from sklearn.model_selection import train_test_split
import numpy as np

class DataLoader:
    def __init__(self):
        self.path = "glaucoma_oct_data/"

    # load the data in folder glaucoma_oct_data/DatasetIEEE
    # see data documentation on OneDrive for "unnamed"
    def ieee_data(self):
        # relevant part is the OD (optic disc) folder for glaucoma data
        folder_structure = os.path.join(self.path,"DatasetIEEE/Dataset/Dataset/OD")
        return 0
    
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
        
        return np_arrays, pathology, patient_id, eye_side


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
    
    # load the data in folder glaucoma_oct_data/OCTandFundusImages
    # see data documentation on OneDrive for "Data on OCT and Fundus Images"
    def oct_fundus(self):
        return 0
    