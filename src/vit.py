from transformers import ViTForImageClassification, ViTFeatureExtractor
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch import nn
import torch
from sklearn.metrics import accuracy_score

# Configuration
VIT_MODEL_NAME = 'google/vit-base-patch16-224'  
IMAGE_SIZE = 224  
NUM_CLASSES = 2  
LR = 0.001  
BATCH_SIZE = 32  
EPOCHS = 10  
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device {DEVICE} to train')

# Data transformations
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


train_dataset = datasets.ImageFolder('../split_data/train', transform=transform)
test_dataset = datasets.ImageFolder('../split_data/test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

vit_model = ViTForImageClassification.from_pretrained(
    VIT_MODEL_NAME, num_labels=NUM_CLASSES, ignore_mismatched_sizes=True).to(DEVICE)

feature_extractor = ViTFeatureExtractor.from_pretrained(VIT_MODEL_NAME)

def train_vit(model, train_loader, epochs=EPOCHS, lr=LR):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images).logits
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader)}")

def test_vit(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images).logits
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

train_vit(vit_model, train_loader)
test_vit(vit_model, test_loader)