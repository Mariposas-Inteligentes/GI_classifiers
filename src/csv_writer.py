import csv

def write_cv_metrics(metrics, model):
    with open(f'../results/cv_{model}_metrics.csv', 'w', newline='') as csvfile:
        fieldnames = ['fold', 'mode', 'accuracy', 'specificity', 'sensitivity']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)

def write_final_results(metrics, model):
    with open(f'../results/{model}_final_results.csv', 'w', newline='') as csvfile:
        fieldnames = ['mode', 'accuracy', 'specificity', 'sensitivity']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)

def write_confusion_matrix(name, matrix):
    with open(f'../results/{name.lower()}_matrix.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(matrix)