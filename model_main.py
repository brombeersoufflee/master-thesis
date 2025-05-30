
import model_implementation as model_impl

model_imp = model_impl.Model_Implementation(model_name="CNNAlex", kfolds=5, augmentation=True)
history = model_imp.train_model()