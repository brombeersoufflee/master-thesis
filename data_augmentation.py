import numpy as np
import random
import cv2 as cv
from scipy.ndimage import gaussian_filter, map_coordinates

class AugmentData:
    def __init__(self, X_train: np.array, y_train: np.array, patient_id: np.array):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.patient_id = patient_id
        self.X_train_augmented = []
        self.y_train_augmented = []
        self.patient_id_augmented = []
    
    def augment_data(self, num_new_volumes: int, return_values= "complete"):
        if num_new_volumes < 0:
            raise ValueError("num_new_volumes must be non-negative")
        if return_values not in ["complete", "augmented"]:
            raise ValueError("return_values must be either 'complete' or 'augmented'")
        
        if num_new_volumes == 0:
            print("No new volumes to generate.")
            return self.X_train, self.y_train, self.patient_id
        if num_new_volumes > 0:
             #init indeces
            brightness_index = 0
            contrast_index = 0
            gamma_index = 0
            gauss_index = 0
            blurring_index = 0
            blurring_sharpening_index = 0
            translation_index = 0
            scaling_index = 0
            deformation_index = 0
            #find all non-glaucomatous instances
            negative_indices = np.where(self.y_train[:, 1] == 0)[0]
            for i in range(0, num_new_volumes):
                random_index = np.random.choice(negative_indices)
                # select a random non-glaucomatous volume and label from the training set
                random_volume = self.X_train[random_index]
                random_label = self.y_train[random_index]
                random_patient_id = self.patient_id[random_index]
                
                # random number for augmentation type and method
                random_method_num = random.random() 
                random_type_num = random.random()

                #TODO: decide which augmentation methods to use and in which proportions
                if random_type_num < 0.5:
                    if random_method_num < 0.2:
                        augmented_volume = self.brightness_change(random_volume)
                        brightness_index +=1
                    elif random_method_num >= 0.2 and random_method_num < 0.4:
                        augmented_volume = self.contrast_change(random_volume)
                        contrast_index +=1
                    elif random_method_num >= 0.4 and random_method_num < 0.6:
                        augmented_volume = self.gamma_correction(random_volume)
                        gamma_index +=1
                # not recommended for the RNFL data augmentation
                # can distort the volume too much
                # elif random_method_num < 0.8:
                #     augmented_volume = self.histogram_equalization(random_volume)
                    elif random_method_num >= 0.6 and random_method_num < 0.8:
                        augmented_volume = self.gaussian_noise(random_volume)
                        gauss_index +=1
                    elif random_method_num >= 0.8 and random_method_num < 0.9:
                        augmented_volume = self.blurring(random_volume)
                        blurring_index +=1
                    else:
                        augmented_volume = self.blurring_then_sharpening(random_volume)
                        blurring_sharpening_index +=1
                elif random_type_num >=0.5 and random_type_num < 0.75:
                    if random_method_num < 0.5: 
                        augmented_volume = self.small_translation(random_volume)
                        translation_index +=1
                    else:
                        augmented_volume = self.mild_scaling(random_volume)
                        scaling_index +=1
                else:
                    augmented_volume = self.elastic_deformation_3d(random_volume)
                    deformation_index +=1
                    # ignore for now maybe apply later in training
                    # elif random_method_num < 1.1:
                    #     augmented_volume = self.cutout(random_volume)
                    # elif random_method_num < 1.15:
                    #     augmented_volume = self.random_erasing(random_volume)
                
                # append the augmented volume to the augmented training set
                self.X_train_augmented.append(augmented_volume)
                self.y_train_augmented.append(random_label)
                self.patient_id_augmented.append(random_patient_id)

        print("Augmented data with the following methods:")
        print(f"Brightness changes: {brightness_index}; {np.round(brightness_index/num_new_volumes*100,2)}%")
        print(f"Contrast changes: {contrast_index}; {np.round(contrast_index/num_new_volumes*100,2)}%")
        print(f"Gamma changes: {gamma_index}; {np.round(gamma_index/num_new_volumes*100,2)}%")
        print(f"Gaussian noise: {gauss_index}; {np.round(gauss_index/num_new_volumes*100,2)}%")
        print(f"Blurring: {blurring_index}; {np.round(blurring_index/num_new_volumes*100,2)}%")
        print(f"Blurring then sharpening: {blurring_sharpening_index}; {np.round(blurring_sharpening_index/num_new_volumes*100,2)}%")
        print(f"Small translation: {translation_index}; {np.round(translation_index/num_new_volumes*100,2)}%")
        print(f"Mild scaling: {scaling_index}; {np.round(scaling_index/num_new_volumes*100,2)}%")
        print(f"Elastic deformation: {deformation_index}; {np.round(deformation_index/num_new_volumes*100,2)}%")
# 
        if return_values == "complete":
            print("Returning both original and augmented data in one.")
            X_train_complete = np.concatenate((self.X_train, self.X_train_augmented), axis=0)
            y_train_complete = np.concatenate((self.y_train, self.y_train_augmented), axis=0)
            patient_id_complete = np.concatenate((self.patient_id, self.patient_id_augmented), axis=0)
            # self.X_train_augmented = None  # Google colab RAM
            # self.y_train_augmented = None # Google colab RAM
            # self.patient_id_augmented = None # Google colab RAM
            return X_train_complete, y_train_complete, patient_id_complete
        if return_values == "augmented":
            print("Returning only augmented data.")
            return self.X_train_augmented, self.y_train_augmented, self.patient_id_augmented
    
    "-------------------------------------------------------------------------------------------------"    
    """The methods below seem reasonable
    - Intensity based augmentations
        - Brightness adjustment
        - Contrast changes
        - Gamma correction
        - Histogram equalization
        - Gaussian noise (to simulate acquisition noise)
        - Blurring/sharpening
    - Affine transfomations (mild)
        - small translations
        - mild scaling
        - elastic deformation (simulate soft tissue variation)
    - cutout/random erasing
        - how would this relate to ROAR/ patch based training"""
    
    # TODO: decide if the factors are good
    
    # brightness modification around the mean with a random shift between -50 and 50
    # https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html
    # applied a brightness shift of 40, the std of the pixel values is around 25 so this creates realistic brightness changes/errors
    # TODO: 
    @staticmethod
    def brightness_change(volume):
        volume = volume.astype(int) #otherwise negative values of the brightness shift throw an error
        brightness_shift = random.randint(-40, 40)
        return np.clip(volume + brightness_shift, 0, 255).astype(np.uint8)
        
    # contrast modification around the mean with a random multiplier between 0.8 and 1.2
    # https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html
    # applying the change around the mean preserves the average brightness of the volume while increasing or decreasing contrast.
    @staticmethod
    def contrast_change(volume):
        contrast_factor = random.uniform(0.5, 1.5)
        mean = np.mean(volume)
        return np.clip((volume - mean) * contrast_factor + mean, 0, 255).astype(np.uint8)
    

    # https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html
    @staticmethod
    def gamma_correction(volume):
        volume = volume.astype(np.uint8)
        gamma = random.uniform(0.75, 1.5)
        # utilize the LUT to speed up the gamma correction calculation (calculte the lookup table once (256 claculations, O(1) after) and use it for all pixels (524.288 pixels per volume))
        lookUpTable = np.empty((1,256))
        for i in range(256):
            lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255).astype(np.uint8)
        return cv.LUT(volume, lookUpTable)
    
    # This probably should not be applied!!!
    # https://docs.opencv.org/3.4/d4/d1b/tutorial_histogram_equalization.html
    @staticmethod
    def histogram_equalization(volume):
        return np.array([cv.equalizeHist(slice) for slice in volume])
    
    @staticmethod
    def gaussian_noise(volume, mean=0):
        sigma = random.randint(0, 25)
        noise = np.random.normal(mean, sigma, volume.shape)
        noisy_volume = volume.astype(np.float32) + noise
        noisy_volume = np.clip(noisy_volume, 0, 255).astype(np.uint8)
        return noisy_volume

    @staticmethod
    def blurring(volume):
        kernel = np.ones((3, 3), np.float32) / 9
        return cv.filter2D(volume, -1, kernel).astype(np.uint8)
    
    # might be too close the original volume
    @staticmethod
    def blurring_then_sharpening(volume):
        blurred_volume = cv.filter2D(volume, -1, np.ones((3, 3), np.float32) / 9).astype(np.uint8)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv.filter2D(blurred_volume, -1, kernel).astype(np.uint8)
    
    # correct but is this shift sensible?
    @staticmethod
    def small_translation(volume):
        tx = random.randint(-5, 5)
        ty = random.randint(-5, 5)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        array = np.array([cv.warpAffine(slice, M, (slice.shape[1], slice.shape[0])).astype(np.uint8) for slice in volume ])
        return array
    
    #correct and scaling values are reasonable
    @staticmethod
    def mild_scaling(volume):
        scale = random.uniform(0.9, 1.1)
    
        center_x = volume.shape[0] // 2
        center_y = volume.shape[1] // 2

        M = np.float32([
            [scale, 0, (1 - scale) * center_x],
            [0, scale, (1 - scale) * center_y]
        ])

        # M = np.float32([[scale, 0, 0], [0, scale, 0]])
        scaled_volume = np.array([cv.warpAffine(slice, M, (volume.shape[0],volume.shape[1])) for slice in volume])
        return scaled_volume.astype(np.uint8)
        
    # TODO:understand the code
    # correct and good values for the distortion
    # alpha could be higher for the RNFL but not for the optic nerve (too many distortions in those 10 slices)
    @staticmethod
    def elastic_deformation_3d(volume, alpha=10, sigma=2, random_state=None):
        """
        Apply elastic deformation to a 3D volume.
        
        Args:
            volume (ndarray): 3D array with shape (D, H, W)
            alpha (float): Scaling factor for the intensity of the deformation
            sigma (float): Standard deviation of the Gaussian filter (controls smoothness)
            random_state (int or None): Seed for reproducibility
        
        Returns:
            ndarray: Deformed 3D volume of same shape, dtype=np.uint8
        """
        if random_state is None:
            random_state = np.random.RandomState(None)
        elif isinstance(random_state, int):
            random_state = np.random.RandomState(random_state)

        shape = volume.shape

        # Generate random displacement fields and smooth them
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        dz = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

        # Create meshgrid of indices
        z, y, x = np.meshgrid(
            np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]),
            indexing='ij'
        )

        # Apply deformation
        indices = np.reshape(z + dz, (-1,)), np.reshape(y + dy, (-1,)), np.reshape(x + dx, (-1,))
        distorted = map_coordinates(volume, indices, order=1, mode='reflect').reshape(shape)

        return np.clip(distorted, 0, 255).astype(np.uint8)


    # TODO: check the below for correctness and document the code
    # This doesn't seem sensible for the data augmentation
    # it could be used for later in training
    @staticmethod
    def cutout(volume):
        h, w = volume.shape[:2]
        mask_size = random.randint(10, 30)
        x = random.randint(0, w - mask_size)
        y = random.randint(0, h - mask_size)
        volume[y:y + mask_size, x:x + mask_size] = 0
        return volume.astype(np.uint8)
    
    @staticmethod
    def random_erasing(volume):
        h, w = volume.shape[:2]
        mask_size = random.randint(10, 30)
        x = random.randint(0, w - mask_size)
        y = random.randint(0, h - mask_size)
        volume[y:y + mask_size, x:x + mask_size] = random.randint(0, 255)
        return volume.astype(np.uint8)

