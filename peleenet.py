from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init as init
from math import sqrt
from itertools import product as product
import torchvision
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
from itertools import product as product

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
INITIALIZE_FROM_SCRATCH = False

class _conv_relu(nn.Module):
    '''
    Simple conv-relu block
    '''

    def __init__(self, in_ch, out_ch, **kwargs):
        super(_conv_relu, self).__init__()

        self.conv = nn.Conv2d(in_ch, out_ch, **kwargs)

        if INITIALIZE_FROM_SCRATCH:
            self.init_weights()    

    def forward(self, input):
        out = F.relu(self.conv(input), inplace=True)
        # print(input.size(), out.size())
        return out
    
    def init_weights(self):
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

class _stem_block(nn.Module):
    '''
    This represents initial stem block that receives input
    '''

    def __init__(self, num_input_ch=3, num_init_features=32):
        super(_stem_block, self).__init__()

        self.stem1 = _conv_relu(num_input_ch, num_init_features, kernel_size=3, stride=2, padding=1, bias=True)
        self.stem2a = _conv_relu(num_init_features, int(num_init_features/2), kernel_size=1, stride=1, padding=0, bias=True)
        self.stem2b = _conv_relu(int(num_init_features/2), num_init_features, kernel_size=3, stride=2, padding=1, bias=True)
        self.stem_pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.stem3 = _conv_relu(2*num_init_features, num_init_features, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, input):
        out = self.stem1(input)

        branch1 = self.stem_pool(out)

        branch2 = self.stem2a(out)
        branch2 = self.stem2b(branch2)

        concat1 = torch.cat([branch1, branch2], dim=1)
        out = self.stem3(concat1)

        return out

class _dense_layer(nn.Module):
    '''
    Individual dense layer
    '''
    def __init__(self, num_input_features, growth_rate, bottleneck_width):
        super(_dense_layer, self).__init__()


        growth_rate = int(growth_rate / 2)
        inter_channel = int(growth_rate  * bottleneck_width / 4) * 4 

        if inter_channel > num_input_features / 2:
            inter_channel = int(num_input_features / 8) * 4
            print('adjust inter_channel to ',inter_channel)

        self.branch1 = nn.Sequential()

        self.branch1.add_module("branch1a", _conv_relu(num_input_features, inter_channel, kernel_size=1, bias=True)) 
        self.branch1.add_module("branch1b", _conv_relu(inter_channel, growth_rate, kernel_size=3, padding=1, bias=True))

        self.branch2 = nn.Sequential()
        self.branch2.add_module("branch2a", _conv_relu(num_input_features, inter_channel, kernel_size=1, bias=True))
        self.branch2.add_module("branch2b", _conv_relu(inter_channel, growth_rate, kernel_size=3, padding=1, bias=True))
        self.branch2.add_module("branch2c", _conv_relu(growth_rate, growth_rate, kernel_size=3, padding=1, bias=True))

    def forward(self, input):
        branch1_out = self.branch1(input)
        branch2_out = self.branch2(input)

        return torch.cat([input, branch1_out, branch2_out], 1)
    
class _dense_block(nn.Sequential):
    '''
    Handles individual denseblock, there are total 4 such dense_block in base pelee
    '''
    def __init__(self,num_dense_layers, num_inp_features, 
                bn_width, growth_rate):
        super(_dense_block, self).__init__()

        for i in range(num_dense_layers):
            nin = num_inp_features + i * growth_rate
            layer = _dense_layer(nin, growth_rate, bn_width)
            self.add_module('dense_layer_%d'%(i+1), layer)
    
class ResBlock(nn.Module):
    """ResBlock that gives 2% boost as mentioned in the paper"""

    def __init__(self, in_channels):
        super(ResBlock, self).__init__()

        self.res1a = nn.Conv2d(in_channels, 128, kernel_size=1)
        self.res1b = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.res1c = nn.Conv2d(128, 256, kernel_size=1)

        self.res2a = nn.Conv2d(in_channels, 256, kernel_size=1)

        if INITIALIZE_FROM_SCRATCH:
            self.init_weights()

    def forward(self, x):
        out1 = F.relu(self.res1a(x),inplace=True)
        out1 = F.relu(self.res1b(out1),inplace=True)
        out1 = self.res1c(out1) #note the one before EltWise layer doesn't have a ReLU in original caffe network

        out2 = self.res2a(x)
        out = out1 + out2
        return out
    
    def init_weights(self):
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)
            

class PeleeBackbone(nn.Module):
    '''
    Backbone stem + densblocks
    '''

    def __init__(self, cfg=None):
        super(PeleeBackbone, self).__init__()

        self.stem_block = nn.Sequential(OrderedDict(
         [('stem_block',_stem_block())]   
        ))

        num_features = cfg['num_init_features']

        #stage1
        stage = 0
        num_dense_layers = cfg['block_config'][stage]
        self.stage1 = _dense_block(num_dense_layers, num_inp_features = num_features, 
                            bn_width = cfg['bottleneck_width'][stage], growth_rate=cfg['growth_rate'][stage])
        
        num_features = num_features + num_dense_layers * cfg['growth_rate'][stage]
        # print("num_features: ", num_features)
        self.stage1_tb = _conv_relu(num_features, num_features, kernel_size=1, stride=1, padding=0)
        self.stage1_tb_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        #stage2
        stage = 1
        num_dense_layers = cfg['block_config'][stage]
        self.stage2 = _dense_block(num_dense_layers, num_inp_features = num_features, 
                            bn_width = cfg['bottleneck_width'][stage], growth_rate=cfg['growth_rate'][stage])
        
        num_features = num_features + num_dense_layers * cfg['growth_rate'][stage]
        # print("num_features: ", num_features)
        self.stage2_tb = _conv_relu(num_features, num_features, kernel_size=1, stride=1, padding=0)
        self.stage2_tb_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        #stage3
        stage = 2
        num_dense_layers = cfg['block_config'][stage]
        self.stage3 = _dense_block(num_dense_layers, num_inp_features = num_features, 
                            bn_width = cfg['bottleneck_width'][stage], growth_rate=cfg['growth_rate'][stage])
        
        num_features = num_features + num_dense_layers * cfg['growth_rate'][stage]
        # print("num_features: ", num_features)
        self.stage3_tb = _conv_relu(num_features, num_features, kernel_size=1, stride=1, padding=0)
        self.stage3_tb_pool = nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True)

        #stage4
        stage = 3
        num_dense_layers = cfg['block_config'][stage]
        self.stage4 = _dense_block(num_dense_layers, num_inp_features = num_features, 
                            bn_width = cfg['bottleneck_width'][stage], growth_rate=cfg['growth_rate'][stage])
        
        num_features = num_features + num_dense_layers * cfg['growth_rate'][stage]
        # print("num_features: ", num_features)
        self.stage4_tb = _conv_relu(num_features, num_features, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        stem_out = self.stem_block(x)

        s1_out = self.stage1(stem_out)
        s1_out = self.stage1_tb(s1_out)
        s1_out = self.stage1_tb_pool(s1_out)

        s2_out = self.stage2(s1_out)
        s2_out = self.stage2_tb(s2_out)
        s2_out = self.stage2_tb_pool(s2_out)

        s3_out = self.stage3(s2_out)
        s3_out_tb = self.stage3_tb(s3_out)
        s3_out = self.stage3_tb_pool(s3_out_tb)

        s4_out = self.stage4(s3_out)
        s4_out_tb = self.stage4_tb(s4_out)

        return s3_out_tb, s4_out_tb

class ExtraBlock(nn.Module):
    '''
    ExtraBlock to downscale further to get 19x19, 10x10, 5x5, 1x1 feature scales used by Resblock
    Note the feature scale size is valid only when input is 304x304
    '''
    def __init__(self, in_ch, out1_ch, last_ext=False):
        super(ExtraBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_ch, out1_ch, kernel_size=1, stride=1, padding=0)
        if not last_ext:
            self.conv2 = nn.Conv2d(out1_ch, 256, kernel_size=3, stride=2,padding=1)
        else:
            self.conv2 = nn.Conv2d(out1_ch, 256, kernel_size=3, stride=2,padding=0)
    
        if INITIALIZE_FROM_SCRATCH:
            self.init_weights()
    def forward(self, x):
        out = F.relu(self.conv1(x), inplace=True)
        out = F.relu(self.conv2(out), inplace=True)

        return out

    def init_weights(self):
        """
        Initialize convolution parameters.
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)


class PeleeNet(nn.Module):
    '''
    Main class for creating the Pelee model.
    args:
        phase: TODO
        size: TODO
        cfg: A dict or a tuple of below
            growth_rate (int or list of 4 ints) - how many filters to add each layer (`k` in paper)
            block_config (list of 4 ints) - how many layers in each pooling block
            num_init_features (int) - the number of filters to learn in the first convolution layer
            bottleneck_width (int or list of 4 ints) - multiplicative factor for number of bottle neck layers
            (i.e. bn_size * k features in the bottleneck layer)
            num_classes (int) - number of classification classes
    '''

    def __init__(self, phase, size, cfg=None):
        super(PeleeNet, self).__init__()

        self.phase = phase
        self.size = size
        self.cfg = cfg
        self.n_classes = self.cfg["num_classes"]

        self.backbone = PeleeBackbone(cfg)

        self.ext1 = ExtraBlock(704, 256)
        self.ext2 = ExtraBlock(256, 128)
        self.ext3 = ExtraBlock(256, 128, last_ext=True)

        self.resblock_1 = ResBlock(512)
        self.resblock_2 = ResBlock(704)
        self.resblock_3 = ResBlock(256)
        self.resblock_4 = ResBlock(256)
        self.resblock_5 = ResBlock(256)

        self.loc = nn.ModuleList()
        self.conf = nn.ModuleList()

        for i, in_ch in enumerate([256] * 6):
            n = cfg["anchor_config"]["anchor_nums"][i] #6 predictions to match with priorbox/anchorbox
            self.loc.append(nn.Conv2d(in_ch, n * 4, kernel_size=1))
            self.conf.append(nn.Conv2d(in_ch, n * cfg["num_classes"], kernel_size=1))

        # if self.phase == 'test':
        #     self.softmax = nn.Softmax(dim=-1)

        self.priors_cxcy = self.create_prior_boxes(cfg)
        # self.priors_cxcy = self.create_prior_boxes()
        # print(self.priors_cxcy.shape)

    def forward(self, x):
        scale_1, scale_2 = self.backbone(x) #19x19, 10x10

        scale_3 = self.ext1(scale_2) #5x5
        scale_4 = self.ext2(scale_3) #3x3
        scale_5 = self.ext3(scale_4) #1x1

        # print("scales", scale_1.shape, scale_2.shape, scale_3.shape, scale_4.shape, scale_5.shape)

        res1_out = self.resblock_1(scale_1)
        res2_out = self.resblock_2(scale_2)
        res3_out = self.resblock_3(scale_3)
        res4_out = self.resblock_4(scale_4)
        res5_out = self.resblock_5(scale_5)

        locs, confs = self.get_loc_conf(res1_out, res2_out, res3_out, res4_out, res5_out)

        # print(locs.shape, confs.shape)

        output = (locs, confs)

        return output

    def get_loc_conf(self, res1_out, res2_out, res3_out, res4_out, res5_out):

        loc1_out = self.loc[0](res1_out).permute(0,2,3,1).contiguous() #equivalent to operating on 19x19
        loc1_out = loc1_out.view(loc1_out.size(0), -1, 4)

        loc2_out = self.loc[1](res1_out).permute(0,2,3,1).contiguous() #equivalent to operating on 19x19 but with different min_max size
        loc2_out = loc2_out.view(loc2_out.size(0), -1, 4)


        loc3_out = self.loc[2](res2_out).permute(0,2,3,1).contiguous() #equivalent to operating on 10x10
        loc3_out = loc3_out.view(loc3_out.size(0), -1, 4)

        loc4_out = self.loc[3](res3_out).permute(0,2,3,1).contiguous() #equivalent to operating on 5x5
        loc4_out = loc4_out.view(loc4_out.size(0), -1, 4)

        loc5_out = self.loc[4](res4_out).permute(0,2,3,1).contiguous() #equivalent to operating on 3x3
        loc5_out = loc5_out.view(loc5_out.size(0), -1, 4)

        loc6_out = self.loc[5](res5_out).permute(0,2,3,1).contiguous() #equivalent to operating on 1x1
        loc6_out = loc6_out.view(loc6_out.size(0), -1, 4)



        conf1_out = self.conf[0](res1_out).permute(0,2,3,1).contiguous() #equivalent to operating on 19x19
        conf1_out = conf1_out.view(conf1_out.size(0), -1, self.cfg["num_classes"])

        conf2_out = self.conf[1](res1_out).permute(0,2,3,1).contiguous() #equivalent to operating on 19x19 but with different min_max size
        conf2_out = conf2_out.view(conf2_out.size(0), -1, self.cfg["num_classes"])

        conf3_out = self.conf[2](res2_out).permute(0,2,3,1).contiguous() #equivalent to operating on 10x10
        conf3_out = conf3_out.view(conf3_out.size(0), -1, self.cfg["num_classes"])

        conf4_out = self.conf[3](res3_out).permute(0,2,3,1).contiguous() #equivalent to operating on 5x5
        conf4_out = conf4_out.view(conf4_out.size(0), -1, self.cfg["num_classes"])

        conf5_out = self.conf[4](res4_out).permute(0,2,3,1).contiguous() #equivalent to operating on 3x3
        conf5_out = conf5_out.view(conf5_out.size(0), -1, self.cfg["num_classes"])

        conf6_out = self.conf[5](res5_out).permute(0,2,3,1).contiguous() #equivalent to operating on 1x1
        conf6_out = conf6_out.view(conf6_out.size(0), -1, self.cfg["num_classes"])

        locs = torch.cat([loc1_out, loc2_out, loc3_out, loc4_out, loc5_out, loc6_out], dim=1)
        confs = torch.cat([conf1_out, conf2_out, conf3_out, conf4_out, conf5_out, conf6_out], dim=1)

        return locs, confs
    
    def create_prior_boxes(self, cfg):
        '''
        Creates prior_boxes in cx,cy,width,height normalized format
        Refer - prior_box_layer.cpp in original ssd caffe : https://github.com/weiliu89/caffe/tree/ssd AND
                prior_box.py in https://github.com/amdegroot/ssd.pytorch
        '''
        _offset = cfg['anchor_config']['offset']
        _min_sizes = cfg['anchor_config']['min_sizes']
        _max_sizes = cfg['anchor_config']['max_sizes']
        _feature_maps = cfg['anchor_config']['feature_maps']
        img_w = cfg['anchor_config']['img_size'][0]
        img_h = cfg['anchor_config']['img_size'][1]
        steps = cfg['anchor_config']['steps']

        prior_boxes = []
        flip_ar = True #can be made configurable

        mean = []
        for k, f in enumerate(_feature_maps):
            for i, j in product(range(int(f)), repeat=2):
                f_k = img_h / steps[k]
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                s_k = _min_sizes[k]/img_h
                mean += [cx, cy, s_k, s_k]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (_max_sizes[k]/img_h))
                mean += [cx, cy, s_k_prime, s_k_prime]

                
                # rest of aspect ratios
                for ar in cfg['anchor_config']['aspect_ratios'][k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]

        # back to torch land
        output = torch.Tensor(mean).view(-1, 4).to(device)
        if True:
            output.clamp_(max=1, min=0)
        return output
    
    def detect_objects(self, predicted_locs, predicted_scores, min_score, max_overlap, top_k):
        """
        Decipher the 8732 locations and class scores (output of ths SSD300) to detect objects.

        For each class, perform Non-Maximum Suppression (NMS) on boxes that are above a minimum threshold.

        :param predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, 8732, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
        :param min_score: minimum threshold for a box to be considered a match for a certain class
        :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via NMS
        :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
        :return: detections (boxes, labels, and scores), lists of length batch_size
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        predicted_scores = F.softmax(predicted_scores, dim=2)  # (N, 8732, n_classes)

        # Lists to store final predicted boxes, labels, and scores for all images
        all_images_boxes = list()
        all_images_labels = list()
        all_images_scores = list()

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        for i in range(batch_size):
            # Decode object coordinates from the form we regressed predicted boxes to
            decoded_locs = cxcy_to_xy(
                gcxgcy_to_cxcy(predicted_locs[i], self.priors_cxcy))  # (8732, 4), these are fractional pt. coordinates

            # Lists to store boxes and scores for this image
            image_boxes = list()
            image_labels = list()
            image_scores = list()

            max_scores, best_label = predicted_scores[i].max(dim=1)  # (8732)

            # Check for each class
            for c in range(1, self.n_classes):
                # Keep only predicted boxes and scores where scores for this class are above the minimum score
                class_scores = predicted_scores[i][:, c]  # (8732)
                score_above_min_score = class_scores > min_score  # torch.uint8 (byte) tensor, for indexing
                n_above_min_score = score_above_min_score.sum().item()
                if n_above_min_score == 0:
                    continue
                class_scores = class_scores[score_above_min_score]  # (n_qualified), n_min_score <= 8732
                class_decoded_locs = decoded_locs[score_above_min_score]  # (n_qualified, 4)

                # Sort predicted boxes and scores by scores
                class_scores, sort_ind = class_scores.sort(dim=0, descending=True)  # (n_qualified), (n_min_score)
                class_decoded_locs = class_decoded_locs[sort_ind]  # (n_min_score, 4)

                # Find the overlap between predicted boxes
                overlap = find_jaccard_overlap(class_decoded_locs, class_decoded_locs)  # (n_qualified, n_min_score)

                # Non-Maximum Suppression (NMS)

                # A torch.uint8 (byte) tensor to keep track of which predicted boxes to suppress
                # 1 implies suppress, 0 implies don't suppress
                suppress = torch.zeros((n_above_min_score), dtype=torch.uint8).to(device)  # (n_qualified)

                # Consider each box in order of decreasing scores
                for box in range(class_decoded_locs.size(0)):
                    # If this box is already marked for suppression
                    if suppress[box] == 1:
                        continue

                    # Suppress boxes whose overlaps (with this box) are greater than maximum overlap
                    # Find such boxes and update suppress indices
                    condition = overlap[box] > max_overlap
                    condition = torch.tensor(condition, dtype=torch.uint8).to(device)
                    suppress = torch.max(suppress, condition)
                    # The max operation retains previously suppressed boxes, like an 'OR' operation

                    # Don't suppress this box, even though it has an overlap of 1 with itself
                    suppress[box] = 0

                # Store only unsuppressed boxes for this class
                image_boxes.append(class_decoded_locs[1 - suppress])
                image_labels.append(torch.LongTensor((1 - suppress).sum().item() * [c]).to(device))
                image_scores.append(class_scores[1 - suppress])

            # If no object in any class is found, store a placeholder for 'background'
            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
                image_labels.append(torch.LongTensor([0]).to(device))
                image_scores.append(torch.FloatTensor([0.]).to(device))

            # Concatenate into single tensors
            image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
            image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
            image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
            n_objects = image_scores.size(0)

            # Keep only the top k objects
            if n_objects > top_k:
                image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k]  # (top_k)
                image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
                image_labels = image_labels[sort_ind][:top_k]  # (top_k)

            # Append to lists that store predicted boxes and scores for all images
            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        return all_images_boxes, all_images_labels, all_images_scores  # lists of length batch_size

class MultiBoxLoss(nn.Module):
    """
    The MultiBox loss, a loss function for object detection.

    This is a combination of:
    (1) a localization loss for the predicted locations of the boxes, and
    (2) a confidence loss for the predicted class scores.
    """

    def __init__(self, priors_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1.):
        super(MultiBoxLoss, self).__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

        self.smooth_l1 = nn.L1Loss()
        self.cross_entropy = nn.CrossEntropyLoss(reduce=False)

    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        """
        Forward propagation.

        :param predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, 8732, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
        :param boxes: true  object bounding boxes in boundary coordinates, a list of N tensors
        :param labels: true object labels, a list of N tensors
        :return: multibox loss, a scalar
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(2)
        # print(n_priors, predicted_locs.size(1), predicted_scores.size(1))
        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(device)  # (N, 8732, 4)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(device)  # (N, 8732)

        # For each image
        for i in range(batch_size):
            n_objects = boxes[i].size(0)

            overlap = find_jaccard_overlap(boxes[i],
                                           self.priors_xy)  # (n_objects, 8732)

            # For each prior, find the object that has the maximum overlap
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)  # (8732)

            # We don't want a situation where an object is not represented in our positive (non-background) priors -
            # 1. An object might not be the best object for all priors, and is therefore not in object_for_each_prior.
            # 2. All priors with the object may be assigned as background based on the threshold (0.5).

            # To remedy this -
            # First, find the prior that has the maximum overlap for each object.
            _, prior_for_each_object = overlap.max(dim=1)  # (N_o)

            # Then, assign each object to the corresponding maximum-overlap-prior. (This fixes 1.)
            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(device)

            # To ensure these priors qualify, artificially give them an overlap of greater than 0.5. (This fixes 2.)
            overlap_for_each_prior[prior_for_each_object] = 1.

            # Labels for each prior
            label_for_each_prior = labels[i][object_for_each_prior]  # (8732)
            # Set priors whose overlaps with objects are less than the threshold to be background (no object)
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0  # (8732)

            # Store
            true_classes[i] = label_for_each_prior

            # Encode center-size object coordinates into the form we regressed predicted boxes to
            true_locs[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_for_each_prior]), self.priors_cxcy)  # (8732, 4)

        # Identify priors that are positive (object/non-background)
        positive_priors = true_classes != 0  # (N, 8732)

        # LOCALIZATION LOSS

        # Localization loss is computed only over positive (non-background) priors
        loc_loss = self.smooth_l1(predicted_locs[positive_priors], true_locs[positive_priors])  # (), scalar

        # Note: indexing with a torch.uint8 (byte) tensor flattens the tensor when indexing is across multiple dimensions (N & 8732)
        # So, if predicted_locs has the shape (N, 8732, 4), predicted_locs[positive_priors] will have (total positives, 4)

        # CONFIDENCE LOSS

        # Confidence loss is computed over positive priors and the most difficult (hardest) negative priors in each image
        # That is, FOR EACH IMAGE,
        # we will take the hardest (neg_pos_ratio * n_positives) negative priors, i.e where there is maximum loss
        # This is called Hard Negative Mining - it concentrates on hardest negatives in each image, and also minimizes pos/neg imbalance

        # Number of positive and hard-negative priors per image
        n_positives = positive_priors.sum(dim=1)  # (N)
        n_hard_negatives = self.neg_pos_ratio * n_positives  # (N)

        # First, find the loss for all priors
        conf_loss_all = self.cross_entropy(predicted_scores.view(-1, n_classes), true_classes.view(-1))  # (N * 8732)
        conf_loss_all = conf_loss_all.view(batch_size, n_priors)  # (N, 8732)

        # We already know which priors are positive
        conf_loss_pos = conf_loss_all[positive_priors]  # (sum(n_positives))

        # Next, find which priors are hard-negative
        # To do this, sort ONLY negative priors in each image in order of decreasing loss and take top n_hard_negatives
        conf_loss_neg = conf_loss_all.clone()  # (N, 8732)
        conf_loss_neg[positive_priors] = 0.  # (N, 8732), positive priors are ignored (never in top n_hard_negatives)
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)  # (N, 8732), sorted by decreasing hardness
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(device)  # (N, 8732)
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)  # (N, 8732)
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]  # (sum(n_hard_negatives))

        # As in the paper, averaged over positive priors only, although computed over both positive and hard-negative priors
        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float()  # (), scalar

        # TOTAL LOSS

        return conf_loss + self.alpha * loc_loss


if __name__ == "__main__":
    writer = SummaryWriter("logs/model_view")
    
    rand_input = torch.randn(3,304,304).unsqueeze(0)

    anchor_config=dict(
            feature_maps=[19, 19, 10, 5, 3, 1],
            steps=[16, 16, 30, 60, 101, 304],
            min_sizes = [21.28, 45.6, 91.2, 136.8, 182.4, 228.0, 273.6], #for 304x304
            max_sizes = [45.6, 91.2, 136.8, 182.4, 228.0, 273.6, 319.2], #for 304x304
            offset = 0.5,
            img_size = (304,304), #W x H
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
            num_classes = 21
        )

    net = PeleeNet("train", (304,304), cfg)
    # ckpt = torch.load("checkpoint_pelee304.pth.tar")
    # model = ckpt['model']
    # torch.save(model.state_dict(), "pelee_304x304_model.pth")

    print(net)
    writer.add_graph(net, rand_input)
    writer.close()
