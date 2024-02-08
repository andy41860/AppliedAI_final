## Applied AI Final Project
## En-Jui Chang, Guo Zhi Wang

from os import listdir, getenv # list filenames in a directory; check running environment
from pandas import read_csv # read CSV files
import numpy as np # array calculation
import time # timer
import matplotlib.pyplot as plt # visualization
import cv2 # image manipulation
import albumentations as A # image augmentation
import torch # neural network
import torch.nn as nn # neural network
import torch.nn.functional as F # neural network
from torch.utils.data import TensorDataset, DataLoader # create dataset and dataloader
import torchvision.transforms as T # transform images
import torchvision.models.segmentation as Seg # pretrained models for segmentation
import torchvision.models as models # pretrained models

if torch.cuda.is_available(): # use GPU if available
    print('GPU name:', torch.cuda.get_device_name(0))
    pu = torch.device('cuda:0')
else: # happy hour(s) with CPU
    pu = torch.device('cpu')

if getenv('COLAB_RELEASE_TAG'): # if running in google colab
    from google.colab import drive # connect to google drive and read files from
    drive.mount('/content/drive')
    PATH = '/content/drive/MyDrive/0 UoL/2 Applied/Assign 3/' # Abby Wang's
else: # runing in local device
    PATH = './'
    
plt.rcParams['figure.constrained_layout.use'] = True # matplotlib display with constrained layout

''' general timer, work as a decorator '''
class Timer():
    def __init__(self, func):
        self.func = func # the target function
    def __call__(self, *args, **kwargs):
        start_time = time.time() # record start time
        return_data = self.func(*args, **kwargs) # run target function
        elapsed_time = time.time() - start_time # get end time and calculate elaspsed time
        print('[%s] elapsed time: %.1f sec' % (self.func.__name__, elapsed_time))
        return return_data # return what target function returned
    
''' read data (image/mask) '''
@Timer
def imread(path): # read images from directory
    list_img, list_msk = [], [] # declare lists of image and mask
    for fn in sorted(listdir(path)): # for all filenames in directory
        f = cv2.imread(path + fn) # read the file
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB) # convert BGR to RGB
        if fn.find('_L') == -1: # if the filename does not contain '_L'
            list_img.append(f) # add the file to image list
        else:
            list_msk.append(f) # otherwise add the file to mask list
    return np.array(list_img), np.array(list_msk) # convert lists to numpy.ndarray and return
train_img, train_msk = imread(PATH + 'train/') # read train data (image/mask)
test_img, test_msk = imread(PATH + 'test/') # read test data (image/mask)
print('Number of train images: %d' % train_img.shape[0])
print('Number of  test images: %d' % test_img.shape[0])
print('Dimensions of image:', train_img.shape[1:])

''' read the color map (definition of color and class) '''
d = read_csv(PATH + 'label_colors.txt', header=None, sep='\t+', engine='python') # read the file
cmap = [[int(c) for c in s.split(' ')] for s in d[0].to_list()] # color map: convert RGB str to int tuple
label = d[1].to_list() # labels
print('Number of classes: %d' % len(label))

''' show some image/mask examples '''
fig, (ax0, ax1) = plt.subplots(2, 2, figsize=(8, 6)) # set a 2*2 multi-axes canvas
ax0[0].imshow(train_img[0]) # show first image
ax0[1].imshow(train_msk[0]) # show first mask
ax1[0].imshow(train_img[-1]) # show last image
ax1[1].imshow(train_msk[-1]) # show last mask
[ax.set_axis_off() for ax in (*ax0, *ax1)] # turn all axis off
plt.show() # show

''' image augmentation for segmentation '''
@Timer
def apply_transform(img, msk, transform, multiply=1): # apply transform to multiple image/mask
    t_img, t_msk = [], []
    for _ in range(multiply): # create data 'multiply' times of the original data
        for i in range(img.shape[0]):
            tf = transform(image=img[i], mask=msk[i]) # tranform the image/mask tuple
            t_img.append(tf['image']) # add the transformed image to image list
            t_msk.append(tf['mask']) # add the transformed mask to mask list
    return np.array(t_img), np.array(t_msk) # turn lists into ndarrays and return

augment = A.Compose([ # define the steps of image transformation for augmentation
    A.Rotate(limit=30, crop_border=True, p=0.8), # rotate image at probability 0.8
    A.RandomCrop(height=240, width=320), # crop a random sub-area
    A.HorizontalFlip(p=0.5), # horinzontal flip at probability 0.5
    A.RandomBrightnessContrast(p=0.2), # change brightness and contrast at probability 0.2
])
aug_img, aug_msk = apply_transform(train_img, train_msk, augment, multiply=4) # augmentation
print('Number of augmented images & masks: %d' % aug_img.shape[0])

''' resize all images/masks '''
resize = A.Resize(height=240, width=320) # resize to smaller resolution for better efficiency
train_img, train_msk = apply_transform(train_img, train_msk, resize) # original train data
test_img, test_msk = apply_transform(test_img, test_msk, resize) # original test data
aug_img, aug_msk = apply_transform(aug_img, aug_msk, resize) # augmented data

''' convert masks to one-hot segmentation maps and then to tensors '''
@Timer
def encode_segmap(msk, cmap): # convert mask(s) to one-hot segmentation map(s)
    pix = msk.reshape(-1, 1, 3) # flatten masks
    maps = (pix == cmap).all(-1) # lookup colormap to get one-hot class of each pixel
    maps = maps.reshape(*msk.shape[:-1], len(cmap)) # resume dimensions
    return maps

@Timer
def map_to_tensor(maps): # convert ndarrays of segmentation maps to tensors
    maps_ = torch.tensor(maps, dtype=torch.float32) # convert to tensor
    maps_ = maps_.permute(0, 3, 1, 2) # dimension (N, H, W, C) to (N, C, H, W)
    return maps_

train_map_ = map_to_tensor(encode_segmap(train_msk, cmap)) # original train data
test_map_ = map_to_tensor(encode_segmap(test_msk, cmap)) # original test data
aug_map_ = map_to_tensor(encode_segmap(aug_msk, cmap)) # augmented data

''' convert images to tensors '''
normalize = T.Normalize(mean=(0.485, 0.456, 0.406), 
                        std=(0.229, 0.224, 0.225)) # standardize to ImageNet average
@Timer
def img_to_tensor(imgs): # convert ndarrays of images to tensors and apply normalization
    imgs_ = torch.tensor(imgs, dtype=torch.float32) # convert to tensor
    imgs_ = imgs_.permute(0, 3, 1, 2) # dimension (N, H, W, C) to (N, C, H, W)
    imgs_ /= 255 # scale to [0, 1]
    imgs_ = normalize(imgs_) # apply normalization
    return imgs_

train_img_ = img_to_tensor(train_img) # original train data
test_img_ = img_to_tensor(test_img) # original test data
aug_img_ = img_to_tensor(aug_img) # augmented data

''' combine original and augmented data to create valid and train data, bring to GPU if available '''
@Timer
def shuffle_(imgs_, maps_): # shuffle images/segmaps
    idx = torch.randperm(imgs_.shape[0]) # create a random order of images/segmaps
    imgs_, maps_ = imgs_[idx], maps_[idx] # apply order to images/segmaps

shuffle_(train_img_, train_map_) # shuffle original train data
shuffle_(aug_img_, aug_map_) # shuffle augmented train data

ori_cut = int(train_img_.shape[0] * 0.9) # num of 90% images in original train data
aug_cut = int(aug_img_.shape[0] * 0.9) # num of 90% images in augmented train data
valid_img_ = torch.cat([train_img_[ori_cut:], aug_img_[aug_cut:]]).to(pu) # valid images
valid_map_ = torch.cat([train_map_[ori_cut:], aug_map_[aug_cut:]]).to(pu) # valid segmaps
train_img_ = torch.cat([train_img_[:ori_cut], aug_img_[:aug_cut]]).to(pu) # train images
train_map_ = torch.cat([train_map_[:ori_cut], aug_map_[:aug_cut]]).to(pu) # train segmaps
test_img_ = test_img_.to(pu) # test images
test_map_ = test_map_.to(pu) # test segmaps

''' create train, valid datasets and dataloaders '''
train_ds = TensorDataset(train_img_, train_map_) # train dataset
valid_ds = TensorDataset(valid_img_, valid_map_) # valid dataset
train_dl = DataLoader(train_ds, batch_size=3, shuffle=True) # train dataloader
valid_dl = DataLoader(valid_ds, batch_size=3, shuffle=True) # valid dataloader

''' define training function '''
@Timer
def train(net, n_epoch, train_dl, valid_dl, criterion, optimizer):
    # inputs: model, num of epochs, train dataloader, valid dataloader, loss function, optimizer
    train_loss_all, valid_loss_all = [], [] # list of losses along epochs
    acc_all, iou_all = [], [] # list of metrics along epochs
    for epoch in range(n_epoch): # train in epochs
        train_loss, valid_loss = 0., 0. # losses of this epoch
        acc, iou = 0., 0. # metrics of this epoch
        
        net.train() # turn model to training mode
        for i, (imgs_, maps_) in enumerate(train_dl): # have 1 batch from train dataloader
            optimizer.zero_grad() # zero the gradients
            pred_ = net(imgs_)['out'] # forward pass
            loss = criterion(pred_, maps_) # calculate the loss
            loss.backward() # backward pass
            optimizer.step() # optimization
            train_loss += loss.item() # sum loss
        
        net.eval() # turn model to evaluating (validating) mode
        with torch.no_grad():
            for i, (imgs_, maps_) in enumerate(valid_dl): # have 1 batch from valid dataloader
                pred_ = net(imgs_)['out'] # forward pass
                loss = criterion(pred_, maps_) # calculate the loss
                valid_loss += loss.item() # sum loss
                acc += pixel_acc(pred_, maps_) # sum pixel-wise accuracy
                iou += mean_iou(pred_, maps_) # sum mean IoU
        
        print('[%2d] train_loss: %.3f  valid_loss: %.3f  pixel_acc: %.3f  mean_IoU: %.3f' 
              % (epoch + 1, train_loss / len(train_dl), valid_loss / len(valid_dl), 
                 acc / len(valid_dl), iou / len(valid_dl)))
        train_loss_all.append(train_loss / len(train_dl)) # add losses of this epoch to list
        valid_loss_all.append(valid_loss / len(valid_dl)) # add losses of this epoch to list
        acc_all.append(acc / len(valid_dl)) # add metrics of this epoch to list
        iou_all.append(iou / len(valid_dl)) # add metrics of this epoch to list
    return train_loss_all, valid_loss_all, acc_all, iou_all

def train_plot(*args, **kwargs): # train model and plot the result
    train_loss, valid_loss, acc, iou = train(*args, **kwargs) # call train function and get results
    fig, ax = plt.subplots(2, 1, figsize=(4, 4))
    ax[0].plot(train_loss, label='Train') # train loss
    ax[0].plot(valid_loss, label='Valid') # valid loss
    ax[0].set(title='Loss')
    ax[0].legend()
    ax[1].plot(acc, label='Pixel-wise Acc') # pixel-wise accuracy
    ax[1].plot(iou, label='Inter-over-Union') # mean IoU
    ax[1].set(title='Metrics', ylim=(0, 1))
    ax[1].legend()
    plt.show()

def release(): # release GPU memory (not all, but will do my best)
    # do "del net, optimizer" before calling this function
    if torch.cuda.is_available(): # if using GPU
        with torch.no_grad():
            torch.cuda.empty_cache() # empty cache

''' define metrics functions '''
def pixel_acc(pred, true): # calculate pixel-wise accuracy
    pred = pred.argmax(dim=1) # convert to integer encoding
    true = true.argmax(dim=1) # convert to integer encoding
    n_pixel = np.prod(true.shape) # total number of pixels
    acc = (pred == true).sum() / n_pixel # num of correctly classified pixels / total num of pixels
    return acc.item()

def mean_iou(pred, true): # calculate mean Intersection over Union (IoU)
    n_class = true.shape[1] # num of classes
    pred = pred.argmax(dim=1) # convert to integer encoding
    true = true.argmax(dim=1) # convert to integer encoding
    iou = []
    for c in range(n_class): # for all classes
        union = torch.sum((true == c) | (pred == c)) # union of area of true and pred
        if union == 0: continue # empty classes are skipped
        inter = torch.sum((pred == c) & (true == c)) # intersection area of true and pred
        iou_class = inter / union # IoU of class = inter / union
        iou.append(iou_class) # add IoU of class to list
    iou = torch.mean(torch.stack(iou)) # mean IoU across all non-empty classes
    return iou.item()

''' define loss functions '''
class DiceLoss(nn.Module): # dice coefficient as loss function
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def __call__(self, pred, true):
        pred = F.sigmoid(pred) # apply sigmoid function to predicted segmap
        sum = (pred + true).sum() # sum of true and pred each
        prod = (pred * true).sum() # summed product of true and pred
        dice = 2 * prod / sum # dice_coef of class = 2 * inter / sum
        return 1 - dice # as loss
    
''' model: FCN_ResNet50 (pretrained), CrossEntropyLoss, SGD, 15 epochs '''
net = Seg.fcn_resnet50(weights=Seg.FCN_ResNet50_Weights.DEFAULT) # load model
net.classifier[4] = nn.Conv2d(net.classifier[4].in_channels, out_channels=len(cmap), 
                              kernel_size=1) # adjust output number of last layer
net.to(pu)
criterion = nn.CrossEntropyLoss() # loss function
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9) # optimizer
release() # release GPU memory
train_plot(net, 15, train_dl, valid_dl, criterion, optimizer) # train and plot results

''' model: FCN_ResNet50 (not pretrained), CrossEntropyLoss, SGD, 15 epochs '''
net = Seg.fcn_resnet50() # load model
net.classifier[4] = nn.Conv2d(net.classifier[4].in_channels, out_channels=len(cmap), 
                              kernel_size=1) # adjust output number of last layer
net.to(pu)
criterion = nn.CrossEntropyLoss() # loss function
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9) # optimizer
release() # release GPU memory
train_plot(net, 15, train_dl, valid_dl, criterion, optimizer) # train and plot results

''' model: FCN_ResNet50, CrossEntropyLoss, Adam, 15 epochs '''
net = Seg.fcn_resnet50() # load model
net.classifier[4] = nn.Conv2d(net.classifier[4].in_channels, out_channels=len(cmap), 
                              kernel_size=1) # adjust output number of last layer
net.to(pu)
criterion = nn.CrossEntropyLoss() # loss function
optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999)) # optimizer
release() # release GPU memory
train_plot(net, 15, train_dl, valid_dl, criterion, optimizer) # train and plot results

''' model: FCN_ResNet50, CrossEntropyLoss, RMSprop, 15 epochs '''
net = Seg.fcn_resnet50() # load model
net.classifier[4] = nn.Conv2d(net.classifier[4].in_channels, out_channels=len(cmap), 
                              kernel_size=1) # adjust output number of last layer
net.to(pu)
criterion = nn.CrossEntropyLoss() # loss function
optimizer = torch.optim.RMSprop(net.parameters(), lr=0.01, alpha=0.99) # optimizer
release() # release GPU memory
train_plot(net, 15, train_dl, valid_dl, criterion, optimizer) # train and plot results

''' model: FCN_ResNet101, CrossEntropyLoss, SGD, 15 epochs '''
net = Seg.fcn_resnet101() # load model
net.classifier[4] = nn.Conv2d(net.classifier[4].in_channels, out_channels=len(cmap), 
                              kernel_size=1) # adjust output number of last layer
net.to(pu)
criterion = nn.CrossEntropyLoss() # loss function
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9) # optimizer
release() # release GPU memory
train_plot(net, 15, train_dl, valid_dl, criterion, optimizer) # train and plot results

''' model: FCN_ResNet50, CrossEntropyLoss, SGD, 80 epochs '''
net = Seg.fcn_resnet50() # load model
net.classifier[4] = nn.Conv2d(net.classifier[4].in_channels, out_channels=len(cmap), 
                              kernel_size=1) # adjust output number of last layer
net.to(pu)
criterion = nn.CrossEntropyLoss() # loss function
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9) # optimizer
release() # release GPU memory
train_plot(net, 80, train_dl, valid_dl, criterion, optimizer) # train and plot results
fcn_resnet50_crl_sgd = net

''' model: FCN_ResNet50, DiceLoss, SGD, 80 epochs '''
net = Seg.fcn_resnet50() # load model
net.classifier[4] = nn.Conv2d(net.classifier[4].in_channels, out_channels=len(cmap), 
                              kernel_size=1) # adjust output number of last layer
net.to(pu)
criterion = DiceLoss() # loss function
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9) # optimizer
release() # release GPU memory
train_plot(net, 80, train_dl, valid_dl, criterion, optimizer) # train and plot results
fcn_resnet50_dice_sgd = net

''' model: DeepLabV3_ResNet50, CrossEntropyLoss, SGD, 15 epochs '''
net = Seg.deeplabv3_resnet50() # load model
net.classifier[4] = nn.Conv2d(net.classifier[4].in_channels, out_channels=len(cmap), 
                              kernel_size=1) # adjust output number of last layer
net.to(pu)
criterion = nn.CrossEntropyLoss() # loss function
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9) # optimizer
release() # release GPU memory
train_plot(net, 15, train_dl, valid_dl, criterion, optimizer) # train and plot results

''' model: DeepLabV3_ResNet50, CrossEntropyLoss, Adam, 15 epochs '''
net = Seg.deeplabv3_resnet50() # load model
net.classifier[4] = nn.Conv2d(net.classifier[4].in_channels, out_channels=len(cmap), 
                              kernel_size=1) # adjust output number of last layer
net.to(pu)
criterion = nn.CrossEntropyLoss() # loss function
optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999)) # optimizer
release() # release GPU memory
train_plot(net, 15, train_dl, valid_dl, criterion, optimizer) # train and plot results

''' model: DeepLabV3_ResNet50, CrossEntropyLoss, RMSprop, 15 epochs '''
net = Seg.deeplabv3_resnet50() # load model
net.classifier[4] = nn.Conv2d(net.classifier[4].in_channels, out_channels=len(cmap), 
                              kernel_size=1) # adjust output number of last layer
net.to(pu)
criterion = nn.CrossEntropyLoss() # loss function
optimizer = torch.optim.RMSprop(net.parameters(), lr=0.01, alpha=0.99) # optimizer
release() # release GPU memory
train_plot(net, 15, train_dl, valid_dl, criterion, optimizer) # train and plot results

''' model: LRASPP_MobileNet_V3_Large, CrossEntropyLoss, SGD, 15 epochs '''
net = Seg.lraspp_mobilenet_v3_large() # load model
net.classifier.high_classifier = nn.Conv2d(net.classifier.high_classifier.in_channels, out_channels=len(cmap), kernel_size=1) # adjust output number of last layer
net.classifier.low_classifier = nn.Conv2d(net.classifier.low_classifier.in_channels, out_channels=len(cmap), kernel_size=1) # adjust output number of last layer
net.to(pu)
criterion = nn.CrossEntropyLoss() # loss function
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9) # optimizer
release() # release GPU memory
train_plot(net, 15, train_dl, valid_dl, criterion, optimizer) # train and plot results

''' model: LRASPP_MobileNet_V3_Large, CrossEntropyLoss, Adam, 15 epochs '''
net = Seg.lraspp_mobilenet_v3_large() # load model
net.classifier.high_classifier = nn.Conv2d(net.classifier.high_classifier.in_channels, out_channels=len(cmap), kernel_size=1) # adjust output number of last layer
net.classifier.low_classifier = nn.Conv2d(net.classifier.low_classifier.in_channels, out_channels=len(cmap), kernel_size=1) # adjust output number of last layer
net.to(pu)
criterion = nn.CrossEntropyLoss() # loss function
optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999)) # optimizer
release() # release GPU memory
train_plot(net, 15, train_dl, valid_dl, criterion, optimizer) # train and plot results

''' model: LRASPP_MobileNet_V3_Large, CrossEntropyLoss, RMSprop, 15 epochs '''
net = Seg.lraspp_mobilenet_v3_large() # load model
net.classifier.high_classifier = nn.Conv2d(net.classifier.high_classifier.in_channels, out_channels=len(cmap), kernel_size=1) # adjust output number of last layer
net.classifier.low_classifier = nn.Conv2d(net.classifier.low_classifier.in_channels, out_channels=len(cmap), kernel_size=1) # adjust output number of last layer
net.to(pu)
criterion = nn.CrossEntropyLoss() # loss function
optimizer = torch.optim.RMSprop(net.parameters(), lr=0.01, alpha=0.99) # optimizer
release() # release GPU memory
train_plot(net, 15, train_dl, valid_dl, criterion, optimizer) # train and plot results

''' model: DeepLabV3_ResNet50, CrossEntropyLoss, SGD, 80 epochs '''
net = Seg.deeplabv3_resnet50() # load model
net.classifier[4] = nn.Conv2d(net.classifier[4].in_channels, out_channels=len(cmap), 
                              kernel_size=1) # adjust output number of last layer
net.to(pu)
criterion = nn.CrossEntropyLoss() # loss function
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9) # optimizer
release() # release GPU memory
train_plot(net, 80, train_dl, valid_dl, criterion, optimizer) # train and plot results
deeplabv3_resnet50_crl_sgd = net

''' model: LRASPP_MobileNet_V3_Large, CrossEntropyLoss, SGD, 80 epochs '''
net = Seg.lraspp_mobilenet_v3_large() # load model
net.classifier.high_classifier = nn.Conv2d(net.classifier.high_classifier.in_channels, out_channels=len(cmap), kernel_size=1) # adjust output number of last layer
net.classifier.low_classifier = nn.Conv2d(net.classifier.low_classifier.in_channels, out_channels=len(cmap), kernel_size=1) # adjust output number of last layer
net.to(pu)
criterion = nn.CrossEntropyLoss() # loss function
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9) # optimizer
release() # release GPU memory
train_plot(net, 80, train_dl, valid_dl, criterion, optimizer) # train and plot results
lraspp_mobilenet_v3_large_crl_sgd = net

''' define testing functions '''
def tensor_to_array(tensor): # convert tensors of segmaps to ndarray
    array = tensor.detach().cpu().permute(0, 2, 3, 1).numpy() # dimension (N, C, H, W) to (N, H, W, C)
    return array

def decode_segmap(maps, cmap): # convert one-hot encoding segmentation map(s) to mask(s)
    pix = maps.argmax(axis=-1) # get the class of each pixel
    msk = np.zeros((*pix.shape, 3), dtype='uint8') # create space as the map size
    for c in range(len(cmap)): # for each class
        msk[pix == c] = cmap[c] # apply RGB color to each pixel
    return msk

def test_plot(net): # test model and plot the result
    # inputs: model
    net.eval() # turn model to evaluating mode
    with torch.no_grad():
        pred_ = net(test_img_)['out'] # predict segmaps with test images (N, C, H, W) by trained model
        acc = pixel_acc(pred_, test_map_) # pixel-wise accuracy
        iou = mean_iou(pred_, test_map_) # mean IoU
    print('[test] pixel_acc: %.3f  mean_IoU: %.3f' % (acc, iou))
    samples = [0, 5, 10] # choose some of test data as samples
    pred_msk = decode_segmap(tensor_to_array(pred_[samples]), cmap) # convert predicted segmaps to masks
    true_img = test_img[samples] # true images
    true_msk = test_msk[samples] # true mask
    
    fig, ax = plt.subplots(len(samples), 2, figsize=(8, 3 * len(samples))) # set a N*2 multi-axes canvas
    for i in range(len(samples)):
        ax[i, 0].imshow(true_img[i], alpha=0.7) # show true image as background
        ax[i, 0].imshow(true_msk[i], alpha=0.5) # show true mask on true image
        ax[i, 1].imshow(true_img[i], alpha=0.7) # show true image as background
        ax[i, 1].imshow(pred_msk[i], alpha=0.5) # show predicted mask on true image
        [ax.set_axis_off() for ax in ax[i]] # turn all axis off
    ax[0, 0].set(title='True')
    ax[0, 1].set(title='Predicted')
    plt.show()

''' plot model: FCN_ResNet50, DiceLoss, SGD '''
test_plot(fcn_resnet50_dice_sgd)
del fcn_resnet50_dice_sgd
release() # release GPU memory

''' plot model: FCN_ResNet50, CrossEntropyLoss, SGD '''
test_plot(fcn_resnet50_crl_sgd)
del fcn_resnet50_crl_sgd
release() # release GPU memory

''' plot model: DeepLabV3_ResNet50, CrossEntropyLoss, SGD '''
test_plot(deeplabv3_resnet50_crl_sgd)
del deeplabv3_resnet50_crl_sgd
release() # release GPU memory

''' plot model: LRASPP_MobileNet_V3_Large, CrossEntropyLoss, SGD '''
test_plot(lraspp_mobilenet_v3_large_crl_sgd)
del lraspp_mobilenet_v3_large_crl_sgd
release() # release GPU memory

