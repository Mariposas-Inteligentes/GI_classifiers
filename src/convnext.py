import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torchvision.models import ConvNeXt_Tiny_Weights
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score #TODO(us): Add other metrics


CONVNEXT_VERSION = 'convnext_tiny'  
NUM_CLASSES = 5
PRETRAINED = True  
LR = 0.0001  
BATCH_SIZE = 32  
EPOCHS = 16
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# TODO(us): Same split for both?
train_dataset = datasets.ImageFolder('../split_data/train', transform=transform)
test_dataset = datasets.ImageFolder('../split_data/test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class ConvNeXtModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(ConvNeXtModel, self).__init__()
        self.model = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)  
        self.model.classifier[2] = nn.Linear(self.model.classifier[2].in_features, num_classes)  

    def forward(self, x):
        return self.model(x)


convnext_model = ConvNeXtModel(NUM_CLASSES).to(DEVICE)


def train_convnext(model, train_loader, epochs=EPOCHS, lr=LR):
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
        
        accuracy = accuracy_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds, average='macro') # TODO(us): find which is the best average suited to our problem
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}')
        print(f'\tAccuracy: {accuracy*100:.2f}%')
        print(f'\tRecall: {recall*100:.2f}%')

def test_convnext(model, test_loader):
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
    recall = recall_score(all_labels, all_preds, average='macro')
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Test Recall: {recall * 100:.2f}%")


train_convnext(convnext_model, train_loader)
test_convnext(convnext_model, test_loader)