import torch
import torchvision
import numpy as np
import time

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from imblearn.metrics import specificity_score
from tqdm import tqdm

from vit import VitBase16
from convnext import ConvNeXtTiny
from csv_writer import write_cv_metrics, write_final_results, write_confusion_matrix

'''
Data storage functions ____________________________________________________________
'''

def write_cv_values(values, model):
    results = []
    results.append({'fold':'1', 'mode':'train', 'accuracy':values[0]['train']['accuracy'], 'sensitivity':values[0]['train']['recall'], 'specificity':values[0]['train']['specificity']})
    results.append({'fold':'1', 'mode':'test', 'accuracy':values[0]['test']['accuracy'], 'sensitivity':values[0]['test']['recall'], 'specificity':values[0]['test']['specificity']})

    results.append({'fold':'2', 'mode':'train', 'accuracy':values[1]['train']['accuracy'], 'sensitivity':values[1]['train']['recall'], 'specificity':values[1]['train']['specificity']})
    results.append({'fold':'2', 'mode':'test', 'accuracy':values[1]['test']['accuracy'], 'sensitivity':values[1]['test']['recall'], 'specificity':values[1]['test']['specificity']})

    results.append({'fold':'3', 'mode':'train', 'accuracy':values[2]['train']['accuracy'], 'sensitivity':values[2]['train']['recall'], 'specificity':values[2]['train']['specificity']})
    results.append({'fold':'3', 'mode':'test', 'accuracy':values[2]['test']['accuracy'], 'sensitivity':values[2]['test']['recall'], 'specificity':values[2]['test']['specificity']})

    results.append({'fold':'4', 'mode':'train', 'accuracy':values[3]['train']['accuracy'], 'sensitivity':values[3]['train']['recall'], 'specificity':values[3]['train']['specificity']})
    results.append({'fold':'4', 'mode':'test', 'accuracy':values[3]['test']['accuracy'], 'sensitivity':values[3]['test']['recall'], 'specificity':values[3]['test']['specificity']})

    results.append({'fold':'5', 'mode':'train', 'accuracy':values[4]['train']['accuracy'], 'sensitivity':values[4]['train']['recall'], 'specificity':values[4]['train']['specificity']})
    results.append({'fold':'5', 'mode':'test', 'accuracy':values[4]['test']['accuracy'], 'sensitivity':values[4]['test']['recall'], 'specificity':values[4]['test']['specificity']})

    write_cv_metrics(results, model)

def write_final_values(values, model):
    results = []
    results.append({'mode':'train', 'accuracy':values['train']['accuracy'], 'sensitivity':values['train']['recall'], 'specificity':values['train']['specificity']})
    results.append({'mode':'test', 'accuracy':values['test']['accuracy'], 'sensitivity':values['test']['recall'], 'specificity':values['test']['specificity']})
    write_final_results(results, model)


'''
Training Functions ________________________________________________________________
'''

def train_model(model, train_loader, epochs, lr, device):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    dictionary = {}
    accuracy = 0
    recall = 0
    specificity = 0
    matrix = None

    for epoch in range(epochs):
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        for batch_index, (images, labels) in enumerate(tqdm(train_loader)):
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
        specificity = specificity_score(all_labels, all_preds, average='macro')
        matrix = confusion_matrix(all_labels, all_preds)

        print(f'\tEpoch [{epoch+1}/{epochs}]')
        print(f'\tLoss: {running_loss/len(train_loader):.4f}')
        print(f'\tAccuracy: {accuracy*100:.2f}%')
        print(f'\tRecall: {recall*100:.2f}%')
        print(f'\tSpecificity: {specificity*100:.2f}%')

    dictionary['accuracy'] = accuracy
    dictionary['recall'] = recall
    dictionary['specificity'] = specificity


    return dictionary, matrix

def test_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    dictionary = {}
    matrix = None

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average='macro')
    specificity = specificity_score(all_labels, all_preds, average='macro')
    matrix = confusion_matrix(all_labels, all_preds)

    print(f'Test Accuracy: {accuracy * 100:.2f}%')
    print(f'Test Recall: {recall * 100:.2f}%')
    print(f'Test Specificity: {specificity * 100:.2f}%')

    dictionary['accuracy'] = accuracy
    dictionary['recall'] = recall
    dictionary['specificity'] = specificity

    return dictionary, matrix

'''
ConvNeXt Functions _______________________________________________________________
'''

def train_test_convnext(path, num_classes, device, transform, batch_size, epochs, lr, save_matrix=False):
    convnext_model = ConvNeXtTiny(num_classes).to(device)

    train_dataset = datasets.ImageFolder(f'../cross_splitted/{path}/train', transform=transform)
    test_dataset = datasets.ImageFolder(f'../cross_splitted/{path}/test', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    dictionary = {}

    start_time = time.time()
    dictionary['train'], train_matrix = train_model(convnext_model, train_loader=train_loader, epochs=epochs, lr=lr, device=device)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Training for fold {path} completed in {elapsed_time:.2f} seconds.')

    dictionary['test'], test_matrix = test_model(convnext_model, test_loader=test_loader, device=device)

    if save_matrix:
        print(train_matrix)
        print(test_matrix)
        write_confusion_matrix('convnext_train', train_matrix)
        write_confusion_matrix('convnext_test', test_matrix)

    return dictionary

def cross_validate_convnext(k):
    NUM_CLASSES = 5
    LR = 0.00001  
    BATCH_SIZE = 32  
    EPOCHS = 8
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device {DEVICE} for ConvNeXt')

    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    values = []
    for i in range(1, k+1):
        print(f'\nTraining and Testing fold #{i}')
        value = train_test_convnext(path=f'fold_{i}', num_classes=NUM_CLASSES, device=DEVICE, transform=transform, batch_size=BATCH_SIZE, epochs=EPOCHS, lr=LR)
        values.append(value)

    write_cv_values(values, 'convnext')

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

def normal_convnext_execution():
    NUM_CLASSES = 5
    LR = 0.00001 
    BATCH_SIZE = 32  
    EPOCHS = 8
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device {DEVICE} for ConvNeXt')

    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    values = train_test_convnext(path=f'fold_{4}', num_classes=NUM_CLASSES, device=DEVICE, transform=transform, batch_size=BATCH_SIZE, epochs=EPOCHS, lr=LR, save_matrix=True)
    write_final_values(values, 'convnext')


'''
ViT Functions _______________________________________________________________
'''

def train_test_vit(path, num_classes, device, transform, batch_size, epochs, lr, save_matrix=False):
    vit_model = VitBase16(num_classes=num_classes, device=device).to(device)

    train_dataset = datasets.ImageFolder(f'../cross_splitted/{path}/train', transform=transform)
    test_dataset = datasets.ImageFolder(f'../cross_splitted/{path}/test', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    dictionary = {}

    start_time = time.time()
    dictionary['train'], train_matrix = train_model(vit_model, train_loader=train_loader, epochs=epochs, lr=lr, device=device)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Training for fold {path} completed in {elapsed_time:.2f} seconds.')

    dictionary['test'], test_matrix = test_model(vit_model, test_loader=test_loader, device=device)

    if save_matrix:
        print(train_matrix)
        print(test_matrix)
        write_confusion_matrix('vit_train', train_matrix)
        write_confusion_matrix('vit_test', test_matrix)

    return dictionary


def cross_validate_vit(k):
    IMAGE_SIZE = 224  
    NUM_CLASSES = 5
    LR = 0.000005
    BATCH_SIZE = 32  
    EPOCHS = 5
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
        print(f'\nTraining and Testing fold #{i}')
        value = train_test_convnext(path=f'fold_{i}', num_classes=NUM_CLASSES, device=DEVICE, transform=transform, batch_size=BATCH_SIZE, epochs=EPOCHS, lr=LR)
        values.append(value)

    write_cv_values(values, 'vit')

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

def normal_vit_execution():
    IMAGE_SIZE = 224  
    NUM_CLASSES = 5
    LR = 0.000005
    BATCH_SIZE = 32  
    EPOCHS = 5
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

    values = train_test_vit(path=f'fold_{4}', num_classes=NUM_CLASSES, device=DEVICE, transform=transform, batch_size=BATCH_SIZE, epochs=EPOCHS, lr=LR, save_matrix=True)
    write_final_values(values, 'vit')


'''
Execution ___________________________________________________________________________________
'''

def main():
    # cross_validate_vit(k=5)
    # cross_validate_convnext(k=5)
    # normal_convnext_execution()
    normal_vit_execution()


if __name__ == "__main__":
    main()