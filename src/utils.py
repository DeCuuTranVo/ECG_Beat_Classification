import pandas as pd
import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split


class EarlyStopping:
    """Early stops the training if validation loss and validation accuracy don't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='models/checkpoint.pt', trace_func=print, monitor='val_loss'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print         
            monitor (Mode): If val_loss, stop at maximum mode, else val_accuracy, stop at minimum mode
                            Default: val_loss   
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_acc_max = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.mode = monitor

    def __call__(self, values, model):

        if self.mode == 'val_loss':
            score = -values

            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(values, model)
            elif score < self.best_score + self.delta:
                self.counter += 1
                self.trace_func(
                    f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(values, model)
                self.counter = 0
        else:
            score = values
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(values, model)
            elif score < self.best_score + self.delta:
                self.counter += 1
                self.trace_func(
                    f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(values, model)
                self.counter = 0

    def save_checkpoint(self, values, model):
        '''Saves model when validation loss decrease.'''
        if self.mode == 'val_loss':
            if self.verbose:
                self.trace_func(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {values:.6f}).   Saving model to {self.path}')
            torch.save(model.state_dict(), self.path)
            self.val_loss_min = values
        elif self.mode == 'val_accuracy':
            if self.verbose:
                self.trace_func(
                    f'Validation accuracy increased ({self.val_acc_max:.3f} --> {values:.3f}).  Saving model to {self.path}')
            torch.save(model.state_dict(), self.path)
            self.val_acc_max = values


def preprocess(data_dir, dataset_name): #csv_dir
    """
    Get training dataframe and testing dataframe from image directory and
    csv description file.

    Args:
        data_dir (String): Directory of image data
        csv_dir (String): Directory of csv description file

    Returns:
        df_train (pandas.DataFrame): Data frame of training set
        df_test (pandas.DataFrame):  Data frame of test set
    """
    
    # print(data_dir)
    # print(dataset_name)
    
    TRAIN_CSV_DIR = None
    TEST_CSV_DIR = None
    
    # dataset_name should be "mitbih" or "ptbdb"     
    for filename in os.listdir(data_dir):
        if (dataset_name.lower() in filename) and ("train" in filename):
            TRAIN_CSV_DIR = os.path.join(data_dir, filename)
            
        if (dataset_name.lower() in filename) and ("test" in filename):
            TEST_CSV_DIR = os.path.join(data_dir, filename)
    
    # print("TRAIN_CSV_DIR:", TRAIN_CSV_DIR)
    # print("TEST_CSV_DIR:", TEST_CSV_DIR)
    
    # print(TRAIN_CSV_DIR)
    # print(TEST_CSV_DIR)
    
    df_train = pd.read_csv(TRAIN_CSV_DIR, header=None)
    df_test = pd.read_csv(TEST_CSV_DIR, header=None)

    # print("preprocessing complete")
    return df_train, df_test


def calculate_metrics(out_gt, out_pred):
    """
    Calculate methics for model evaluation

    Args:
        out_gt (torch.Tensor)   : Grouth truth array
        out_pred (torch.Tensor) : Prediction array

    Returns:
        accuracy (float)    : Accuracy
        precision (float)   : Precision
        recall (float)      : Recall
        f1_score (float)    : F1 Score
        sensitivity (float) : Sensitivity
        specificity (float) : Specificity

    """
    true_positives = 0.0
    true_negatives = 0.0
    false_positives = 0.0
    false_negatives = 0.0

    for i in range(len(out_gt)):
        if ((out_gt[i] == 1) and (out_pred[i] == 1)):
            true_positives += 1
        if ((out_gt[i] == 0) and (out_pred[i] == 0)):
            true_negatives += 1
        if ((out_gt[i] == 0) and (out_pred[i] == 1)):
            false_positives += 1
        if ((out_gt[i] == 1) and (out_pred[i] == 0)):
            false_negatives += 1

    accuracy = (true_positives + true_negatives) / (true_positives +
                                                    true_negatives + false_positives + false_negatives)

    precision = true_positives / \
        (true_positives + false_positives + np.finfo(float).eps)
    recall = true_positives / \
        (true_positives + false_negatives + np.finfo(float).eps)

    f1_score = 2 * precision * recall / \
        (precision + recall + np.finfo(float).eps)

    sensitivity = recall
    specificity = true_negatives / \
        (true_negatives + false_positives + np.finfo(float).eps)

    return accuracy, precision, recall, f1_score, sensitivity, specificity


from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

def calculate_metrics(out_gt, out_pred, num_classes):
    """
    Calculate methics for model evaluation

    Args:
        out_gt (torch.Tensor)   : Grouth truth array
        out_pred (torch.Tensor) : Prediction array

    Returns:
        accuracy (float)    : Accuracy
        precision (float)   : Precision
        recall (float)      : Recall
        f1_score (float)    : F1 Score
        sensitivity (float) : Sensitivity
        specificity (float) : Specificity

    """
    
    out_gt = out_gt.cpu().detach().numpy()
    out_pred = out_pred.cpu().detach().numpy()
    
    # print("out_gt", out_gt)
    # print("out_gt shape", out_gt.shape)
    # print("out_pred", out_pred)
    # print("out_pred shape", out_pred.shape)
    
    # # print("out_gt_metrics", out_gt)
    # print("shape of out_gt_metrics",out_gt.shape)
    # # # print(out_gt.dtype)
    # # print("out_pred_metrics",out_pred)
    # print("shape of out_pred_metrics",out_pred.shape)
    # # # print(out_pred.dtype)
    
    accuracy = accuracy_score(out_gt, out_pred)
    precision = precision_score(out_gt, out_pred, average = "macro", zero_division=0)
    recall = recall_score(out_gt, out_pred, average = "macro", zero_division=0)
    F1_score = f1_score(out_gt, out_pred, average = "macro", zero_division=0)

    return accuracy, precision, recall, F1_score


from sklearn.metrics import confusion_matrix , classification_report
import matplotlib.pyplot as plt
import seaborn as sns
def create_confusion_matrix(out_gt, out_pred, problem, trainer):
    out_gt = out_gt.cpu().detach().numpy()
    out_pred = out_pred.cpu().detach().numpy()
    
    # out_gt = to_categorical(out_gt.astype('uint8'), num_classes=4)

    # out_pred = np.argmax(out_pred, axis=1)
    # out_pred = to_categorical(out_pred.astype('uint8'), num_classes=4)
    
    # print(out_gt)
    # print(out_pred)
    # print(np.unique(out_pred, axis=0))
    
    cm = confusion_matrix(out_gt , out_pred, normalize='true')
    # print(cm)
    # print([i for i in np.unique(out_pred, axis=0)])
    
    if problem == "mitbih":
        label_list = ["N", "S", "P", "F", "U"]
        index_list = [0, 1, 2, 3, 4]
        columns_list = [0, 1, 2, 3, 4]
    elif problem == "ptbdb":
        label_list = ["Normal", "Abnormal"]
        index_list = [0, 1]
        columns_list = [0, 1]
    else:
        raise Exception("Only two values are allowed: 'mitbih' and 'ptbdb'")
    
    cm = pd.DataFrame(cm , index = index_list , columns = columns_list)
    # print(cm)
    # ######################################################################################
    cm_figure = plt.figure(figsize = (10,8))    
    cm_heatmap = sns.heatmap(cm, linecolor = 'black' , linewidth = 0.5 , annot = True, xticklabels = label_list, yticklabels =label_list  ) #  fmt='d',
    plt.title('Confusion Matrix', fontsize=20)
    plt.xlabel("Prediction", fontsize=18)
    plt.ylabel("Ground Truth", fontsize=18)
    # plt.show()
    # plt.imshow(cm_heatmap)
    
    plt.savefig(os.path.join("media", "trial-" + trainer.TRIAL + "-confusion_matrix.png"))
    return cm


if __name__ == '__main__':
    pass
