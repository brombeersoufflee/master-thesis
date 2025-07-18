from swin_transformer import SwinTransformer
import torch
import torch.nn as nn
import torch.optim as optim
from monai.utils import ensure_tuple_rep
from torch.utils.data import DataLoader, TensorDataset
from data_load import DataLoader as DataLoader_local

data_loader_local = DataLoader_local()
print("Loading data...")
volumes, labels, patient_id, eye_side = data_loader_local.retina_npy()
print("Splitting data...")
train_volumes, test_volumes, train_labels, test_labels, train_patient_id, test_patient_id, train_eye_side, test_eye_side = data_loader_local.retina_npy_split(volumes, labels, patient_id, eye_side)
print(f"Train data shape: {train_volumes.shape}, Train labels shape: {train_labels.shape}")
print(f"Test data shape: {test_volumes.shape}, Test labels shape: {test_labels.shape}")

train_dataset = TensorDataset(torch.tensor(train_volumes), torch.tensor(train_labels))
# test_dataset = TensorDataset(test_volumes, test_labels)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

patch_size = ensure_tuple_rep(2, 3) # (2,2,2)
window_size = ensure_tuple_rep(7, 3) # (7,7,7)
SwinViT = SwinTransformer(
            in_chans=1,
            embed_dim=96, # args.feature_size 48 # embed is patchsize* channel_size
            window_size=window_size,
            patch_size=patch_size,
            depths=[2, 2, 6, 2], # if 18 then swin-s and if 6 then swin-t
            num_heads=[3, 6, 12, 24],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            norm_layer=torch.nn.LayerNorm,
            spatial_dims=3,
            classification=True,  # Set to True for classification tasks
            )
            # patch_norm: bool = False,
            # use_checkpoint: bool = False,
            # downsample="merging",
            # num_classes: int = 2,

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(SwinViT.parameters(), lr=0.01)

# Dummy data
inputs = torch.randn(1, 1, 10, 10, 10)  # 10 samples, each with 10 features
labels = torch.randint(0, 2, (10,))  # Random integer labels (0 or 1) for 10 samples

# Training loop
SwinViT.train()  # Set the model to training mode
optimizer.zero_grad()  # Zero the gradients
outputs = SwinViT(inputs)  # Forward pass
print("Model Outputs (before softmax):\n", outputs)

loss = criterion(outputs, labels)  # Compute the loss
print("Loss:", loss.item())

loss.backward()  # Backward pass (compute gradients)
optimizer.step()  # Update model parameters

# Updated model parameters for fc1 layer (optional)
print("\nUpdated fc1 weights:\n", SwinViT.fc1.weight)
print("Updated fc1 biases:\n", SwinViT.fc1.bias)