import numpy as np
import random
import cv2 as cv

class AugmentData:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.X_train_augmented = []
        self.y_train_augmented = []
    
    def augment_data(self, num_new_images, return_values= "complete"):
        if num_new_images < 0:
            raise ValueError("num_new_images must be non-negative")
        if return_values not in ["complete", "augmented"]:
            raise ValueError("return_values must be either 'complete' or 'augmented'")
        
        if num_new_images == 0:
            print("No new images to generate.")
            return self.X_train, self.y_train
        if num_new_images > 0:
            for i in range(num_new_images):
                random_index = np.random.randint(0, len(self.X_train))
                random_image = self.X_train[random_index]
                random_label = self.y_train[random_index]
                
                random_method_num = random.random()
                #TODO: decide which augmentation methods to use and in which proportions
                if random_method_num < 0.5:
                    # apply augmentation method 1
                    augmented_image = self.augmentation_method(random_image)
                else:
                    # apply augmentation method 2
                    augmented_image = self.augmentation_method2(random_image)
                
                # append the augmented image to the augmented training set
                self.X_train_augmented.append(augmented_image)
                self.y_train_augmented.append(random_label)

        if return_values == "complete":
            print("Returning both original and augmented data in one.")
            X_train_complete = self.X_train.copy() + self.X_train_augmented.copy()
            y_train_complete = self.y_train.copy() + self.y_train_augmented.copy()
            return X_train_complete, y_train_complete
        if return_values == "augmented":
            print("Returning only augmented data.")
            return self.X_train_augmented, self.y_train_augmented
        
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
    def brightness_change(image):
        image = image.astype(np.int) #otherwise negative values of the brightness shift throw an error
        brightness_shift = random.randint(-40, 40)
        return np.clip(image + brightness_shift, 0, 255).astype(np.uint8)
        
    # contrast modification around the mean with a random multiplier between 0.8 and 1.2
    # https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html
    # applying the change around the mean preserves the average brightness of the image while increasing or decreasing contrast.
    def contrast_change(image):
        contrast_factor = random.uniform(0.5, 1.5)
        mean = np.mean(image)
        return np.clip((image - mean) * contrast_factor + mean, 0, 255).astype(np.uint8)
    

    # https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html
    def gamma_correction(image):
        gamma = random.uniform(0.75, 1.5)
        # utilize the LUT to speed up the gamma correction calculation (calculte the lookup table once (256 claculations, O(1) after) and use it for all pixels (524.288 pixels per image))
        lookUpTable = np.empty((1,256))
        for i in range(256):
            lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255).astype(np.uint8)
        return cv.LUT(image, lookUpTable)
    
    # TODO: check the below for correctness and document the code
    def histogram_equalization(image):
        hist, bins = np.histogram(image.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * (255 / cdf[-1])
        return np.interp(image.flatten(), bins[:-1], cdf_normalized).reshape(image.shape).astype(np.uint8)
    
    def gaussian_noise(image):
        noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
        return np.clip(image + noise, 0, 255).astype(np.uint8)
    
    def blurring(image):
        kernel = np.ones((5, 5), np.float32) / 25
        return cv2.filter2D(image, -1, kernel).astype(np.uint8)
    
    def sharpening(image):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(image, -1, kernel).astype(np.uint8)
    
    def small_translation(image):
        tx = random.randint(-5, 5)
        ty = random.randint(-5, 5)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        return cv2.warpAffine(image, M, (image.shape[1], image.shape[0])).astype(np.uint8)
    
    def mild_scaling(image):
        scale = random.uniform(0.9, 1.1)
        M = np.float32([[scale, 0, 0], [0, scale, 0]])
        return cv2.warpAffine(image, M, (image.shape[1], image.shape[0])).astype(np.uint8)
    
    def elastic_deformation(image):
        random_state = np.random.RandomState(None)
        shape = image.shape
        dx = random_state.rand(*shape) * 2 - 1
        dy = random_state.rand(*shape) * 2 - 1
        dz = random_state.rand(*shape) * 2 - 1
        dz[0] = 0
        dz[-1] = 0
        return np.clip(image + dx + dy + dz, 0, 255).astype(np.uint8)

    def cutout(self, image):
        h, w = image.shape[:2]
        mask_size = random.randint(10, 30)
        x = random.randint(0, w - mask_size)
        y = random.randint(0, h - mask_size)
        image[y:y + mask_size, x:x + mask_size] = 0
        return image.astype(np.uint8)
    
    def random_erasing(self, image):
        h, w = image.shape[:2]
        mask_size = random.randint(10, 30)
        x = random.randint(0, w - mask_size)
        y = random.randint(0, h - mask_size)
        image[y:y + mask_size, x:x + mask_size] = random.randint(0, 255)
        return image.astype(np.uint8)

