import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
# from model import SSD300, 
from peleenet import PeleeNet, MultiBoxLoss
from datasets import PascalVOCDataset
from utils import *

from torch.utils.tensorboard import SummaryWriter
import datetime

log_folder = "./logs/training/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(log_folder)

# Data parameters
data_folder = './'  # folder with data files
keep_difficult = True  # use objects considered difficult to detect?

# Model parameters
# Not too many here since the SSD300 has a very specific structure
n_classes = len(label_map)  # number of different types of objects
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Learning parameters
checkpoint = None  # path to model checkpoint, None if none
batch_size = 8  # batch size
iterations = 120000  # number of iterations to train
workers = 4  # number of workers for loading data in the DataLoader
print_freq = 200  # print training status every __ batches
lr = 1e-3  # learning rate
decay_lr_at = [80000, 100000]  # decay learning rate after these many iterations
decay_lr_to = 0.1  # decay learning rate to this fraction of the existing learning rate
momentum = 0.9  # momentum
weight_decay = 5e-4  # weight decay
grad_clip = None  # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation

cudnn.benchmark = True

anchor_config=dict(
        feature_maps=[19, 19, 10, 5, 3, 1],
        min_sizes = [21.28, 45.6, 91.2, 136.8, 182.4, 228.0, 273.6], #for 304x304
        max_sizes = [45.6, 91.2, 136.8, 182.4, 228.0, 273.6, 319.2], #for 304x304
        offset = 0.5,
        img_size = (304,304), #W x H
        steps=[16, 16, 30, 60, 101, 304],
        min_ratio=15,
        max_ratio=90,
        aspect_ratios=[[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2,3]],
        anchor_nums=[6, 6, 6, 6, 6, 6]
        )

cfg = dict(
        growth_rate = [32]*4,
        block_config=[3, 4, 8, 6],
        num_init_features=32,
        bottleneck_width=[1, 2, 4, 4],
        anchor_config = anchor_config,
        num_classes = n_classes
    )


def main():
    """
    Training.
    """
    global start_epoch, label_map, epoch, checkpoint, decay_lr_at, cfg
    
    # Initialize model or load checkpoint
    if checkpoint is None:
        start_epoch = 0
        model = PeleeNet("train", (304,304), cfg) #SSD300(n_classes=n_classes)
        # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)
        
        optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                                    lr=lr, momentum=momentum, weight_decay=weight_decay)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # Move to default device
    model = model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

    # Custom dataloaders
    train_dataset = PascalVOCDataset(data_folder,
                                     split='train',
                                     keep_difficult=keep_difficult)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=train_dataset.collate_fn, num_workers=workers,
                                               pin_memory=True)  # note that we're passing the collate function here

    # Calculate total number of epochs to train and the epochs to decay learning rate at (i.e. convert iterations to epochs)
    # To convert iterations to epochs, divide iterations by the number of iterations per epoch
    # The paper trains for 120,000 iterations with a batch size of 32, decays after 80,000 and 100,000 iterations
    epochs = iterations // (len(train_dataset) // 32)
    decay_lr_at = [it // (len(train_dataset) // 32) for it in decay_lr_at]

    curr_loss = None
    min_loss = None
    # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate at particular epochs
        if epoch in decay_lr_at:
            adjust_learning_rate(optimizer, decay_lr_to)

        # One epoch's training
        curr_loss = train(train_loader=train_loader,
                    model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    epoch=epoch)

        # Save checkpoint
        save_checkpoint(epoch, model, optimizer)
        if not min_loss:
            min_loss = curr_loss #initialize first time
        
        #save intermediate epochs
        if  curr_loss <= min_loss:
            model_name = "pelee_304_epoch_{0}_loss_{1}.pth".format(epoch, curr_loss)
            model_save_path = os.path.join("~/work/pytorch_ws/pelee_od_pytorch/state_dicts/",model_name)
            torch.save(model.state_dict(), model_save_path)

def train(train_loader, model, criterion, optimizer, epoch):
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()

    # Batches
    for i, (images, boxes, labels, _) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to default device
        images = images.to(device)  # (batch_size (N), 3, 300, 300)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        # Forward prop.
        predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

        # Loss
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # Update model
        optimizer.step()

        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))
            writer.add_scalar('training loss', losses.avg, epoch * len(train_loader) + i)
    del predicted_locs, predicted_scores, images, boxes, labels  # free some memory since their histories may be stored
    return losses.avg


if __name__ == '__main__':
    main()
