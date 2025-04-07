import os
import SimpleITK as sitk

class DataLoader:
    def __init__(self):
        self.a=0

    # load the data in folder glaucoma_oct_data/DatasetIEEE
    # see data documentation on drive for "unnamed"
    def ieee_data(self):
        # relevant part is the OD folder for glaucoma data
        folder_structure = "glaucoma_oct_data/DatasetIEEE/Dataset/Dataset/OD"
        return 0
    
    # load the data in folder glaucoma_oct_data/retina-oct-glaucoma
    # see data documentation on drive for "Retina OCT Glaucoma dataset"
    def retina(self):
        path = "glaucoma_oct_data/retina-oct-glaucoma/retina-oct-glaucoma/imagesTr"
        images = []
        np_arrays = []

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
    # see data documentation on drive for "Data on OCT and Fundus Images"
    def oct_fundus(self):
        return 0
    