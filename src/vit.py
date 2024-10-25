import torch
import torchvision

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import nn
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score #TODO(us): Add other metrics

IMAGE_SIZE = 224  
NUM_CLASSES = 5
LR = 0.001  
BATCH_SIZE = 32  
EPOCHS = 10  
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device {DEVICE}')

transform = transforms.Compose(
    [
        transforms.Resize(256, interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_dataset = datasets.ImageFolder('../split_data/train', transform=transform)
test_dataset = datasets.ImageFolder('../split_data/test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class VitBase16(nn.Module):
    def __init__(self, num_classes, device):
        super(VitBase16, self).__init__()
        pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
        self.vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights).to(device)

        for parameter in self.vit.parameters():
            parameter.requires_grad = False

        self.vit.heads = nn.Linear(in_features=768, out_features=num_classes).to(device)

    def forward(self, x):
        return self.vit(x)

vit_model = VitBase16(num_classes=NUM_CLASSES, device=DEVICE).to(DEVICE)

def train_vit(model, train_loader, epochs=EPOCHS, lr=LR):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")
        accuracy = accuracy_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds, average='macro') # TODO(us): find which is the best average suited to our problem
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}')
        print(f'\tAccuracy: {accuracy*100:.2f}%')
        print(f'\tRecall: {recall*100:.2f}%')

def test_vit(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

train_vit(vit_model, train_loader)
test_vit(vit_model, test_loader)