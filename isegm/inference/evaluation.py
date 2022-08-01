from time import time

import numpy as np
import torch

from isegm.inference import utils
from isegm.inference.clicker import Clicker

try:
    get_ipython()
    from tqdm import tqdm_notebook as tqdm
except NameError:
    from tqdm import tqdm


def evaluate_dataset(dataset, predictor, max_clicks, **kwargs):
    all_ious = []
    all_ious_np = np.zeros((len(dataset), max_clicks))
    all_ious_np.fill(np.nan)
    all_dice_scores = np.zeros((len(dataset), max_clicks))
    all_dice_scores.fill(np.nan)

    start_time = time()
    for index in tqdm(range(len(dataset)), leave=False):
        sample = dataset.get_sample(index)

        _, sample_ious, _, sample_dice_scores = evaluate_sample(sample.image, sample.gt_mask, predictor,
                                            sample_id=index, max_clicks=max_clicks, **kwargs)
        all_ious_np[index, 0:len(sample_ious)] = sample_ious
        all_dice_scores[index, 0:len(sample_dice_scores)] = sample_dice_scores
        all_ious.append(sample_ious)

    end_time = time()
    elapsed_time = end_time - start_time

    return all_ious, elapsed_time, all_ious_np, all_dice_scores


def evaluate_sample(image, gt_mask, predictor, max_iou_thr,
                    pred_thr=0.49, min_clicks=1, max_clicks=20,
                    sample_id=None, callback=None):
    clicker = Clicker(gt_mask=gt_mask)
    pred_mask = np.zeros_like(gt_mask)
    ious_list = []
    dice_score_list = []

    with torch.no_grad():
        predictor.set_input_image(image)

        for click_indx in range(max_clicks):
            clicker.make_next_click(pred_mask)
            pred_probs = predictor.get_prediction(clicker)
            pred_mask = pred_probs > pred_thr

            if callback is not None:
                callback(image, gt_mask, pred_probs, sample_id, click_indx, clicker.clicks_list)

            iou, dice_score = utils.get_iou(gt_mask, pred_mask)
            ious_list.append(iou)
            dice_score_list.append(dice_score)

            if iou >= max_iou_thr and click_indx + 1 >= min_clicks:
                break
        return clicker.clicks_list, np.array(ious_list, dtype=np.float32), pred_probs, \
               np.array(dice_score_list, dtype=np.float32)
