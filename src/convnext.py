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
EPOCHS = 2
REPETITIONS = 5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device {DEVICE}')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


class ConvNeXtModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(ConvNeXtModel, self).__init__()
        self.model = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)  
        self.model.classifier[2] = nn.Linear(self.model.classifier[2].in_features, num_classes)  

    def forward(self, x):
        return self.model(x)


def train_convnext(model, train_loader, epochs=EPOCHS, lr=LR):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    dictionary = {}
    accuracy = 0
    recall = 0

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

    dictionary['accuracy'] = accuracy
    dictionary['recall'] = recall

    return dictionary

def test_convnext(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []

    dictionary = {}
    
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
    dictionary['accuracy'] = accuracy
    dictionary['recall'] = recall

    return dictionary

convnext_model = ConvNeXtModel(NUM_CLASSES).to(DEVICE)

def train_test(path):
    train_dataset = datasets.ImageFolder(f'../cross_splitted/{path}/train', transform=transform)
    test_dataset = datasets.ImageFolder(f'../cross_splitted/{path}/test', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    dictionary = {}

    dictionary['train'] = train_convnext(convnext_model, train_loader)
    dictionary['test'] = test_convnext(convnext_model, test_loader)

    return dictionary


values = []
for i in range(1, 6):
    print(f'Training and Testing #{i}')
    values.append(train_test(f'fold_{i}'))


metrics = ['accuracy', 'recall']

average_train = [0] * len(metrics)
average_test = [0] * len(metrics)

for j in range(len(metrics)):
    for i in range(len(values)):    
        average_train[j] += values[i]['train'][metrics[j]]
        average_test[j] += values[i]['test'][metrics[j]]
    average_train[j] /= REPETITIONS
    average_test[j] /= REPETITIONS
    print(f'\nAverage {metrics[j]} for training: {average_train[j]}')
    print(f'\nAverage {metrics[j]} for testing: {average_test[j]}')
