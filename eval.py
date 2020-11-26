from utils import *
from datasets import PascalVOCDataset
from tqdm import tqdm
from pprint import PrettyPrinter
from peleenet import PeleeNet, MultiBoxLoss

# Good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()

# Parameters
data_folder = './'
keep_difficult = True  # difficult ground truth objects must always be considered in mAP calculation, because these objects DO exist!
batch_size = 64
workers = 4

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
        num_classes = 21
    )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
checkpoint = './pelee_304x304_model.pth'

# Load model checkpoint that is to be evaluated

model = PeleeNet("test",(304,304),cfg)
checkpoint = torch.load(checkpoint, map_location=device)
model.load_state_dict(checkpoint)
# model = checkpoint['model']
model = model.to(device)

# Switch to eval mode
model.eval()

# Load test data
test_dataset = PascalVOCDataset(data_folder,
                                split='test',
                                keep_difficult=keep_difficult)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                          collate_fn=test_dataset.collate_fn, num_workers=workers, pin_memory=True)


def evaluate(test_loader, model):
    """
    Evaluate.

    :param test_loader: DataLoader for test data
    :param model: model
    """

    # Make sure it's in eval mode
    model.eval()

    # Lists to store detected and true boxes, labels, scores
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    true_difficulties = list()  # it is necessary to know which objects are 'difficult', see 'calculate_mAP' in utils.py

    with torch.no_grad():
        # Batches
        for i, (images, boxes, labels, difficulties) in enumerate(tqdm(test_loader, desc='Evaluating')):
            images = images.to(device)  # (N, 3, 300, 300)

            # Forward prop.
            predicted_locs, predicted_scores = model(images)

            # Detect objects in SSD output
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores,
                                                                                       min_score=0.01, max_overlap=0.45,
                                                                                       top_k=200)
            # Evaluation MUST be at min_score=0.01, max_overlap=0.45, top_k=200 for fair comparision with the paper's results and other repos

            # Store this batch's results for mAP calculation
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            difficulties = [d.to(device) for d in difficulties]

            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.extend(boxes)
            true_labels.extend(labels)
            true_difficulties.extend(difficulties)

        # Calculate mAP
        APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties)

    # Print AP for each class
    pp.pprint(APs)

    print('\nMean Average Precision (mAP): %.3f' % mAP)


if __name__ == '__main__':
    evaluate(test_loader, model)
