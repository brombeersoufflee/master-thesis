import os
import SimpleITK as sitk

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
        path = os.path.join(self.path,"retina-oct-glaucoma/retina-oct-glaucoma/imagesTr")
        json_path = os.path.join(self.path,"retina-oct-glaucoma/retina-oct-glaucoma/dataset.json")
        images = []
        np_arrays = []

        # Load JSON label info
        with open(json_path, "r") as f:
            metadata = json.load(f)
            label_data = metadata["training"]

        for filename in os.listdir(path):
            if filename.endswith(".mha"):
                full_path = os.path.join(path, filename)
                image = sitk.ReadImage(full_path)
                images.append(image)
                np_arrays.append(sitk.GetArrayFromImage(image))
            else:
                print("non-mha object found:",filename)
        
        return images, np_arrays
    
    # load the data in folder glaucoma_oct_data/OCTandFundusImages
    # see data documentation on OneDrive for "Data on OCT and Fundus Images"
    def oct_fundus(self):
        return 0
    