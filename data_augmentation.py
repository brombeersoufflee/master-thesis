import numpy as np
import random

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
                #TODO: extend plethora of augmentation methods
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
        

    def augmentation_method():
        return None
    
    def augmentation_method2():
        return None

