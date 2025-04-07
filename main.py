import data_load

data_loader = data_load.DataLoader()
image_data, np_array_data = data_loader.retina()

print(len(image_data))

print(image_data[0])
print(np_array_data[0])