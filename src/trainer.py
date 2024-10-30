import torch
import torchvision
import numpy as np
import csv

from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import nn
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

from vit import VitBase16
from convnext import ConvNeXtTiny


'''
General Functions __________________________________________________________
'''

def macro_specificity_score(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred)
    specificity_per_class = []

    for i in range(len(matrix)):
        tn = np.sum(np.delete(np.delete(matrix, i, axis=0), i, axis=1))
        fp = np.sum(matrix[:, i]) - matrix[i, i]
        specificity = tn / (tn + fp)
        specificity_per_class.append(specificity)

    macro_specificity = np.mean(specificity_per_class)
    return macro_specificity

def append_results_to_csv(file_path, k, y_true, y_pred):
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        for true_value, pred_value in zip(y_true, y_pred):
            writer.writerow([k, true_value, pred_value])

'''
ViT Functions _______________________________________________________________
'''

def train_vit(model, train_loader, epochs, lr, device):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    
    dictionary = {}

    for epoch in range(epochs):
        running_loss = 0.0
        all_preds = []
        all_labels = []

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
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
        recall = recall_score(all_labels, all_preds, average='macro')
        specificity = macro_specificity_score(all_labels, all_preds)

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}')
        print(f'\tAccuracy: {accuracy*100:.2f}%')
        print(f'\tRecall: {recall*100:.2f}%')
        print(f'\tSpecificity: {specificity*100:.2f}%')

    dictionary['accuracy'] = accuracy
    dictionary['recall'] = recall
    dictionary['specificity'] = specificity

    return dictionary, all_preds, all_labels

def test_vit(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    dictionary = {}

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average='macro')
    specificity = macro_specificity_score(all_labels, all_preds)

    print(f'Test Accuracy: {accuracy * 100:.2f}%')
    print(f'Test Recall: {recall * 100:.2f}%')
    print(f'Test Specificity: {specificity * 100:.2f}%')

    dictionary['accuracy'] = accuracy
    dictionary['recall'] = recall
    dictionary['specificity'] = specificity
    
    return dictionary, all_preds, all_labels

def train_test_vit(path, num_classes, device, transform, batch_size, epochs, lr, k):
    vit_model = VitBase16(num_classes=num_classes, device=device).to(device)

    train_dataset = datasets.ImageFolder(f'../cross_splitted/{path}/train', transform=transform)
    test_dataset = datasets.ImageFolder(f'../cross_splitted/{path}/test', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    dictionary = {}
    dictionary['train'], all_preds, all_labels = train_vit(vit_model, train_loader, device=device, epochs=epochs, lr=lr)
    append_results_to_csv('../results/vit_training.csv', k, all_labels, all_preds)

    dictionary['test'], all_preds, all_labels = test_vit(vit_model, test_loader, device=device)
    append_results_to_csv('../results/vit_testing.csv', k, all_labels, all_preds)

    return dictionary

def cross_validate_vit(k):
    IMAGE_SIZE = 224  
    NUM_CLASSES = 5
    LR = 0.0001  
    BATCH_SIZE = 32  
    EPOCHS = 1
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device {DEVICE} for ViT')

    transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    values = []
    for i in range(1, k+1):
        print(f'Training and Testing fold #{i}')
        value = train_test_vit(path=f'fold_{i}', num_classes=NUM_CLASSES, device=DEVICE, transform=transform, batch_size=BATCH_SIZE, epochs=EPOCHS, lr=LR, k=i)
        values.append(value)

    metrics = ['accuracy', 'recall', 'specificity']
    average_train = [0] * len(metrics)
    average_test = [0] * len(metrics)

    for j in range(len(metrics)):
        for i in range(len(values)):
            average_train[j] += values[i]['train'][metrics[j]]
            average_test[j] += values[i]['test'][metrics[j]]
        average_train[j] /= k
        average_test[j] /= k
        print(f'\nAverage {metrics[j]} for training: {average_train[j]}')
        print(f'\nAverage {metrics[j]} for testing: {average_test[j]}')


'''
ConvNeXt Functions ____________________________________________________________
'''

def train_convnext(model, train_loader, epochs, lr, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    

    dictionary = {}
    accuracy = 0
    recall = 0
    specificity = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
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
        recall = recall_score(all_labels, all_preds, average='macro')
        specificity = macro_specificity_score(all_labels, all_preds)

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}')
        print(f'\tAccuracy: {accuracy*100:.2f}%')
        print(f'\tRecall: {recall*100:.2f}%')
        print(f'Specificity: {specificity*100:.2f}%')

    dictionary['accuracy'] = accuracy
    dictionary['recall'] = recall
    dictionary['specificity'] = specificity

    return dictionary

def test_convnext(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    dictionary = {}
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average='macro')
    specificity = macro_specificity_score(all_labels, all_preds)

    print(f'Test Accuracy: {accuracy * 100:.2f}%')
    print(f'Test Recall: {recall * 100:.2f}%')
    print(f'Test Specificity: {specificity * 100:.2f}%')

    dictionary['accuracy'] = accuracy
    dictionary['recall'] = recall
    dictionary['specificity'] = specificity

    return dictionary

def train_test_convnext(path, num_classes, device, transform, batch_size, epochs, lr):
    convnext_model = ConvNeXtTiny(num_classes).to(device)

    train_dataset = datasets.ImageFolder(f'../cross_splitted/{path}/train', transform=transform)
    test_dataset = datasets.ImageFolder(f'../cross_splitted/{path}/test', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    dictionary = {}
    dictionary['train'] = train_convnext(convnext_model, train_loader=train_loader, epochs=epochs, lr=lr, device=device)
    dictionary['test'] = test_convnext(convnext_model, test_loader=test_loader, device=device)

    return dictionary

def cross_validate_convnext(k):
    NUM_CLASSES = 5
    LR = 0.0001  
    BATCH_SIZE = 32  
    EPOCHS = 10
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device {DEVICE} for ConvNeXt')

    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])


    values = []
    for i in range(1, k+1):
        print(f'\nTraining and Testing fold #{i}')
        value = train_test_convnext(path=f'fold_{i}', num_classes=NUM_CLASSES, device=DEVICE, transform=transform, batch_size=BATCH_SIZE, epochs=EPOCHS, lr=LR)
        values.append(value)

    metrics = ['accuracy', 'recall', 'specificity']

    average_train = [0] * len(metrics)
    average_test = [0] * len(metrics)

    for j in range(len(metrics)):
        for i in range(len(values)):
            average_train[j] += values[i]['train'][metrics[j]]
            average_test[j] += values[i]['test'][metrics[j]]
        average_train[j] /= k
        average_test[j] /= k
        print(f'\nAverage {metrics[j]} for training: {average_train[j]}')
        print(f'\nAverage {metrics[j]} for testing: {average_test[j]}')

'''
Execution
'''

def main():
    # cross_validate_vit(k=5)
    cross_validate_convnext(k=1)


if __name__ == "__main__":
    main()