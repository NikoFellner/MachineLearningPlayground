import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToTensor
from Models import Autoencoder


plot_ = False
dir_ = "data"

training_data = datasets.FashionMNIST(
    root=dir_,
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root=dir_,
    train=False,
    download=True,
    transform=ToTensor()
)

if plot_:
    labels_map = {
        0: "T-Shirt",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot",
    }
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(training_data), size=(1,)).item()
        img, label = training_data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Autoencoder().to(device)
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 7

for epoch in range(num_epochs):
    for images, _ in train_dataloader:  # We ignore labels since it's unsupervised
        images = images.to(device)  # Move to GPU if available

        # Forward pass (Encode -> Decode)
        outputs = model(images)
        loss = criterion(outputs, images)  # Compare reconstructed image with input

        # Backpropagation
        optimizer.zero_grad()  # Reset gradients
        loss.backward()  # Compute gradients
        optimizer.step()  # Update weights

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

test_images, _ = next(iter(test_dataloader))
test_images = test_images[:5].to(device)

# Pass through autoencoder
reconstructed = model(test_images).cpu().detach()

# Plot original and reconstructed images
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i in range(5):
    axes[0, i].imshow(test_images[i].cpu().squeeze(), cmap="gray")
    axes[0, i].axis("off")
    axes[1, i].imshow(reconstructed[i].squeeze(), cmap="gray")
    axes[1, i].axis("off")

axes[0, 0].set_title("Original Images")
axes[1, 0].set_title("Reconstructed Images")
plt.show()