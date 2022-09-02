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
    all_dices = []

    start_time = time()
    for index in tqdm(range(len(dataset)), leave=False):
        sample = dataset.get_sample(index)

        _, sample_ious, _ = evaluate_sample(sample.image, sample.gt_mask, predictor, metric='iou',
                                            sample_id=index, max_clicks=max_clicks, **kwargs)
        _, sample_dices, _ = evaluate_sample(sample.image, sample.gt_mask, predictor, metric='dice',
                                             sample_id=index, max_clicks=max_clicks, **kwargs)
        all_ious.append(sample_ious)
        all_dices.append(sample_dices)

    end_time = time()
    elapsed_time = end_time - start_time

    return all_ious, all_dices, elapsed_time


def evaluate_sample(image, gt_mask, predictor, max_metric_thr, metric,
                    pred_thr=0.49, min_clicks=1, max_clicks=20,
                    sample_id=None, callback=None):
    clicker = Clicker(gt_mask=gt_mask)
    pred_mask = np.zeros_like(gt_mask)
    metric_list = []

    with torch.no_grad():
        predictor.set_input_image(image)

        for click_indx in range(max_clicks):
            clicker.make_next_click(pred_mask)
            pred_probs = predictor.get_prediction(clicker)
            pred_mask = pred_probs > pred_thr

            if callback is not None and sample_id % 1000 == 0:
                callback(image, gt_mask, pred_probs, sample_id, click_indx, clicker.clicks_list)

            if metric == 'iou':
                metric_value = utils.get_iou(gt_mask, pred_mask)
            elif metric == 'dice':
                metric_value = utils.get_dice(gt_mask, pred_mask)
            else:
                print(f"Metric {metric} is not implemented")
            metric_list.append(metric_value)

            if metric_value >= max_metric_thr and click_indx + 1 >= min_clicks:
                break
        return clicker.clicks_list, np.array(metric_list, dtype=np.float32), pred_probs
