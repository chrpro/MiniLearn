import torch
import torchvision
import copy

import pandas as pd
from os import path
from typing import Dict
from torch import nn
from scipy import stats
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from PIL import Image

# from torchvision.transforms import Compose, Normalize, ToTensor

from tqdm import tqdm
TT   = transforms.ToTensor()
NRM  = transforms.Normalize( (0.5), (0.5)) 
TPIL = transforms.ToPILImage()
transform_no_aug   = transforms.Compose( [ TPIL ] )
# transform_no_aug   = transforms.Compose( [ ] )
import os
nnscript = os.path.abspath('../../scripts')
os.sys.path.append(nnscript)
from nnom import *
from mfcc import *


def mfcc_plot(x, label= None):
    mfcc_feat = np.swapaxes(x, 0, 1)
    ig, ax = plt.subplots()
    cax = ax.imshow(mfcc_feat, interpolation='nearest', origin='lower', aspect=1)
    if label is not None:
        ax.set_title(label)
    else:
        ax.set_title('MFCC')
    plt.show()

def label_to_category(label, selected):
    category = []
    for word in label:
        if(word in selected):
            category.append(selected.index(word))
        else:
            category.append(len(selected)) # all others
    return np.array(category)



def normalize(data, n, quantize=True):
    limit = pow(2, n)
    data = np.clip(data, -limit, limit) / limit
    if quantize:
        data = np.round(data * 128) / 128.0
    return data


# load data
try:
    x_train = np.load('train_data.npy')
    y_train = np.load('train_label.npy')
    x_test = np.load('test_data.npy')
    y_test = np.load('test_label.npy')
    x_val = np.load('val_data.npy')
    y_val = np.load('val_label.npy')
except:
    (x_train, y_train), (x_test, y_test), (x_val, y_val) = merge_mfcc_file()
    np.save('train_data.npy', x_train)
    np.save('train_label.npy', y_train)
    np.save('test_data.npy', x_test)
    np.save('test_label.npy', y_test)
    np.save('val_data.npy', x_val)
    np.save('val_label.npy', y_val)

selected_lable = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight','five', 'follow', 'forward',
                      'four','go','happy','house','learn','left','marvin','nine','no','off','on','one','right',
                      'seven','sheila','six','stop','three','tree','two','up','visual','yes','zero']


num_type = len(selected_lable)+1

x_train = x_train[:, :, 1:]
x_test = x_test[:, :, 1:]
x_val = x_val[:, :, 1:]
# expand on channel axis because we only have one channel
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))
x_val = x_val.reshape((x_val.shape[0], x_val.shape[1], x_val.shape[2], 1))
# print('x_train shape:', x_train.shape, 'max', x_train.max(), 'min', x_train.min())

# training data enforcement
x_train = np.vstack((x_train, x_train*0.8))
y_train = np.hstack((y_train, y_train))
# print(y_train.shape)


# instead of using maximum value for quantised, we allows some saturation to save more details in small values.
x_train = normalize(x_train, 3)
x_test = normalize(x_test, 3)
x_val = normalize(x_val, 3)

# print('quantised', 'x_train shape:', x_train.shape, 'max', x_train.max(), 'min', x_train.min())
# print("dataset abs mean at", abs(x_test).mean()*128)


    # test, if you want to see a few random MFCC imagea. 
# if(1):
#     which = 232
#     while True:
#         mfcc_plot(x_train[which].reshape((62, 12))*128, y_train[which])
#         which += 3

y_train, _ = pd.factorize( y_train) 
y_test, _ = pd.factorize( y_test) 
y_val, _ = pd.factorize( y_val)

# y_train = [[i] for i in y_train]
# print(np.unique(y_train))
# y_test = label_to_category(y_test, selected_lable)
# y_val = label_to_category(y_val, selected_lable)

# print(y_test)

# word label to number label
# y_train = label_to_category(y_train, selected_lable)
# y_test = label_to_category(y_test, selected_lable)
# y_val = label_to_category(y_val, selected_lable)

# number label to onehot
# y_train = tf.keras.utils.to_categorical(y_train, num_type)
# y_test = tf.keras.utils.to_categorical(y_test, num_type)
# y_val = tf.keras.utils.to_categorical(y_val, num_type)

# shuffle test data
permutation = np.random.permutation(x_test.shape[0])
x_test = x_test[permutation, :]
y_test = y_test[permutation]
permutation = np.random.permutation(x_train.shape[0])
x_train = x_train[permutation, :]
y_train = y_train[permutation]

# print(type(x_train))
# # generate test data for MCU
# generate_test_bin(x_test * 127, y_test, 'test_data.h')
# generate_test_bin(x_train * 127, y_train, 'train_data.h')
validation_ratio = 0.10
test_ratio = 0.10
x_val, x_train, y_val, y_train = train_test_split(x_train, y_train, test_size=test_ratio/(test_ratio + validation_ratio), shuffle=False) 


# Define a function to separate classes by class index
def get_class_i(x, y, i):
    """
    x: trainset.train_data or testset.test_data
    y: trainset.train_labels or testset.test_labels
    i: class label, a number between 0 to 9
    return: x_i
    """
    # Convert to a numpy array
    y = np.array(y)
    # Locate position of labels that equal to i
    pos_i = np.argwhere(y == i)
    # Convert the result into a 1-D list
    pos_i = list(pos_i[:,0])
    # Collect all data that match the desired label
    x_i = [x[j] for j in pos_i]
    
    return x_i


class MyDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = torch.LongTensor(targets)
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        
        if self.transform:
            x = Image.fromarray(self.data[index].astype(np.uint8).transpose(1,2,0))
            x = self.transform(x)
        
        return x, y
    
    def __len__(self):
        return len(self.data)


class DatasetMaker(Dataset):
    def __init__(self, datasets, transformFunc = transform_no_aug):
        """
        datasets: a list of get_class_i outputs, i.e. a list of list of images for selected classes
        """
        self.datasets = datasets
        self.lengths  = [len(d) for d in self.datasets]
        self.transformFunc = transformFunc
    def __getitem__(self, i):
        class_label, index_wrt_class = self.index_of_which_bin(self.lengths, i)
        img = self.datasets[class_label][index_wrt_class]
        img = self.transformFunc(img)
        return img, class_label

    def __len__(self):
        return sum(self.lengths)
    
    def index_of_which_bin(self, bin_sizes, absolute_index, verbose=False):
        """
        Given the absolute index, returns which bin it falls in and which element of that bin it corresponds to.
        """
        # Which class/bin does i fall into?
        accum = np.add.accumulate(bin_sizes)
        if verbose:
            print("accum =", accum)
        bin_index  = len(np.argwhere(accum <= absolute_index))
        if verbose:
            print("class_label =", bin_index)
        # Which element of the fallent class/bin does i correspond to?
        index_wrt_class = absolute_index - np.insert(accum, 0, 0)[bin_index]
        if verbose:
            print("index_wrt_class =", index_wrt_class)

        return bin_index, index_wrt_class


def sample_from_class(data_set, k):
    """
    function to sample data and their labels from a dataset in pytorch in
    a stratified manner
    Args
    ----
    data_set
    k: the number of samples that will be accuimulated in the new slit
    Returns
    -----
    train_dataset
    val_dataset
    """
    class_counts = {}
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    for data, label in data_set:
        class_i = label.item() if isinstance(label, torch.Tensor) else label
        class_counts[class_i] = class_counts.get(class_i, 0) + 1
        if class_counts[class_i] <= k:
            train_data.append(data)
            train_label.append(label)
        else:
            test_data.append(data)
            test_label.append(label)

    train_data = torch.stack(train_data)
    train_label = torch.tensor(train_label, dtype=torch.int64)
    test_data = torch.stack(test_data)
    test_label = torch.tensor(test_label, dtype=torch.int64)

    return (
        TensorDataset(train_data, train_label),
        TensorDataset(test_data, test_label),
    )


def load_har():
    # num_labels = 6
    # train_ratio = 0.80

    # train is now 75% of the entire data set
    # x_train, x_test, y_train, y_test = train_test_split(reshaped_segments, labels, test_size=1 - train_ratio)
    # # test is now 10% of the initial data set validation is now 10% of the initial data set
    # x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio), shuffle=False) 


    # sub_trainset = \
    #     DatasetMaker(
    #         [ 
    #          get_class_i(x_train, y_train, 0), #:0
    #          get_class_i(x_train, y_train, 1), #:1
    #          get_class_i(x_train, y_train, 2), #:2
    #         ],
    #         transform_no_aug
    #     )
    # sub_testset  = \
    #     DatasetMaker(
    #         [
    #          get_class_i(x_test, y_test, 0), 
    #          get_class_i(x_test, y_test, 1), 
    #          get_class_i(x_test, y_test, 2), 
    #         ],
    #         transform_no_aug
    #     )
 
    # transform = transforms.Compose([transforms.ToTensor()])
    # sub_trainset = MyDataset(x_train, y_train, transform=transform)
    # sub_testset = MyDataset(x_test, y_test, transform=transform)
    # sub_valset = MyDataset(x_val, y_val, transform=transform)


    train_data = torch.tensor(x_train)
    train_label = torch.tensor(y_train, dtype=torch.int64)
    test_data = torch.tensor(x_test)
    test_label = torch.tensor(y_test, dtype=torch.int64)
    val_data = torch.tensor(x_val)
    val_label = torch.tensor(y_val, dtype=torch.int64)


    sub_trainset = TensorDataset(train_data, train_label)# transform=transforms.ToTensor() )
    sub_testset = TensorDataset(test_data, test_label)#, transform=transforms.ToTensor() )
    sub_valset = TensorDataset(val_data, val_label)#, transform=transforms.ToTensor() )

    # traindLoader = DataLoader(sub_trainset)
    # testLoader = DataLoader(sub_testset)
    # valLoader = DataLoader(sub_valset)
    

    return [sub_trainset, sub_testset, sub_valset]

    # my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
    # my_dataloader = DataLoader(my_dataset) # create your dataloader


    # return [sub_trainset, sub_valset, sub_testset]

    # sub_trainset2 = \
    #     DatasetMaker(sub_trainset,
    #         transform_no_aug
    #     )
    # sub_testset2 = \
    #     DatasetMaker(sub_testset,
    #         transform_no_aug
    #     )
    # sub_valset2 = \
    #     DatasetMaker(sub_valset,
    #         transform_no_aug
    #     )
    # sub_valset, tr_set = sample_from_class(sub_trainset, 100)

    # return [sub_trainset, sub_valset, sub_testset]
    # return [sub_trainset, sub_valset, sub_testset]

class SampleCNN(nn.Module):
    def __init__(self, shape=(1,62,12), batch_size=32):
        super().__init__()
        self.input_shape = shape

        self.batch_size = batch_size
        # self.dropout1 = nn.Dropout(p=0.2, inplace=False)
        # print(shape)
        self.conv1 = nn.Conv2d(in_channels=shape[0], out_channels=64, kernel_size=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(1)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(1)


        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1)
        self.relu3 = nn.ReLU()
        # self.pool3 = nn.MaxPool2d(2)
        # self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1)
        # self.relu4 = nn.ReLU()


        self.flatten = nn.Flatten()
        self.interface_shape = self.get_shape()
        self.interface = nn.Linear(in_features=self.interface_shape.numel(), out_features=36)

        

    def get_shape(self):
        sample = torch.randn(size=(self.batch_size, *self.input_shape))
        out = self.conv1(sample)
        out = self.relu1(out)
        # out = self.pool1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        # out = self.pool2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        return out.shape[1:]

    def forward(self, x):
        # print(x.shape)
        out = self.conv1(x)
        # out = self.pool1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        # out = self.pool2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = self.flatten(out)
        out = self.relu3(out)
        return self.interface(out)
        # return self.fc(out)


class SimpleTrainer:
    def __init__(
        self, datasets=None, dataloaders=None, models_path=".", cuda="cuda:0",
    ):
        super().__init__()
        self.datasets = datasets
        # TODO: choose GPU with less memory
        self.devicy = torch.device(
            cuda if torch.cuda.is_available() else "cpu"
        )
        self.datasizes = {
            i: len(sett)
            for i, sett in zip(["train", "val", "test"], self.datasets)
        }
        self.models_path = models_path
        self.dataloaders = dataloaders
        self.criterion = nn.CrossEntropyLoss()

    def train(
        self, net: nn.Module, parameters: Dict[str, float], name: str,
    ) -> nn.Module:

        net.to(device=self.devicy)  # pyre-ignore [28]
        optimizer = torch.optim.Adam(
            net.parameters(), lr=parameters.get("learning_rate")
        )
        exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=parameters.get("learning_step"),
            gamma=parameters.get("learning_gamma"),
        )

        # Train Network
        net = self.train_loop(
            net, optimizer, exp_lr_scheduler, name, parameters.get("epochs")
        )
        return net

    def train_loop(self, model, optimizer, scheduler, name, epochs):
        """
        Training loop
        """
        total_acc = []
        best_loss = 10 ** 8
        t = tqdm(range(epochs))
        for _ in t:
            for phase in ["train", "val"]:
                if phase == "train":
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0.0

                for inputs, labels in self.dataloaders[phase]:

                    inputs = inputs.to(self.devicy)
                    labels = labels.to(self.devicy)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)
                        # backward + optimize only if in training phase
                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * labels.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    if phase == "train":
                        scheduler.step()

                epoch_acc = running_corrects / self.datasizes[phase]
                epoch_loss = running_loss / self.datasizes[phase]
                total_acc.append(epoch_acc)
                # deep copy the model
                if phase == "val" and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    t.set_description("Validation accuracy %.3f" % epoch_acc)
                    t.refresh()
        # fig = plt.figure()
        # plt.plot(np.array(total_acc), 'r')
        # plt.savefig('foo.png')
        # plt.close(fig)

        model.load_state_dict(best_model_wts)
        torch.save(
            model.state_dict(), path.join(self.models_path, str(name) + ".pth")
        )
        return model

    def evaluate(self, net: nn.Module) -> float:

        correct = 0
        total = 0
        data_loader = self.dataloaders["test"]
        net.eval()
        with torch.no_grad():

            for inputs, labels in data_loader:
                inputs = inputs.to(device=self.devicy)
                labels = labels.to(device=self.devicy)
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct / total
