import torch
import numpy as np
from tkinter import messagebox
import matplotlib.pyplot as plt

from isegm.inference import clicker
from isegm.inference.predictors import get_predictor
from isegm.utils.vis import draw_with_blend_and_clicks
from isegm.inference.utils import get_iou, get_dice


class InteractiveController:
    def __init__(self, net, device, predictor_params, update_image_callback, prob_thresh=0.5, one_input_channel=False,
                 checkpoint=''):
        self.net = net
        self.prob_thresh = prob_thresh
        self.clicker = clicker.Clicker()
        self.states = []
        self.probs_history = []
        self.object_count = 0
        self._result_mask = None
        self._init_mask = None
        self._ground_truth_mask = None

        self.image = None
        self.predictor = None
        self.device = device
        self.update_image_callback = update_image_callback
        self.predictor_params = predictor_params
        self.reset_predictor()
        self.one_input_channel = one_input_channel
        self.dice = [0.0]
        self.iou = [0.0]
        self.clicks = [0]
        self.thresholds = [0.0]
        self.n_clicks = 0
        self.checkpoint = checkpoint

    def set_image(self, image):
        if self.one_input_channel:
            self.image = image[:, :, 0]
            self.image = self.image[:, :, None]
        else:
            self.image = image

        self._result_mask = np.zeros(image.shape[:2], dtype=np.uint16)
        self.object_count = 0
        self.reset_last_object(update_image=False)
        self.update_image_callback(reset_canvas=True)

        self.dice = [0.0]
        self.iou = [0.0]

    def set_mask(self, mask):
        if self.image.shape[:2] != mask.shape[:2]:
            messagebox.showwarning("Warning", "A segmentation mask must have the same sizes as the current image!")
            return

        if len(self.probs_history) > 0:
            self.reset_last_object()

        self._init_mask = mask.astype(np.float32)
        self.probs_history.append((np.zeros_like(self._init_mask), self._init_mask))
        self._init_mask = torch.tensor(self._init_mask, device=self.device).unsqueeze(0).unsqueeze(0)
        self.clicker.click_indx_offset = 1

        if self._ground_truth_mask is not None:
            self.calculate_score(mask)

    def set_ground_truth_mask(self, mask):
        if self.image.shape[:2] != mask.shape[:2]:
            messagebox.showwarning("Warning", "A segmentation mask must have the same sizes as the current image!")
            return

        self._ground_truth_mask = mask / 255

    def add_click(self, x, y, is_positive):
        self.states.append({
            'clicker': self.clicker.get_state(),
            'predictor': self.predictor.get_states()
        })

        click = clicker.Click(is_positive=is_positive, coords=(y, x))
        self.clicker.add_click(click)
        self.n_clicks += 1
        pred = self.predictor.get_prediction(self.clicker, prev_mask=self._init_mask)
        if self._init_mask is not None and len(self.clicker) == 1:
            pred = self.predictor.get_prediction(self.clicker, prev_mask=self._init_mask)

        torch.cuda.empty_cache()

        if self.probs_history:
            self.probs_history.append((self.probs_history[-1][0], pred))
        else:
            self.probs_history.append((np.zeros_like(pred), pred))

        self.update_image_callback(save_values=True)

    def undo_click(self):
        if not self.states:
            return

        prev_state = self.states.pop()
        self.clicker.set_state(prev_state['clicker'])
        self.predictor.set_states(prev_state['predictor'])
        self.probs_history.pop()
        if not self.probs_history:
            self.reset_init_mask()
        self.update_image_callback()

        self.dice = self.dice[:-1]
        self.iou = self.iou[:-1]
        self.clicks = self.clicks[:-1]
        self.thresholds = self.thresholds[:-1]
        self.n_clicks -= 1
        self.calculate_score(mask1=None, print_only=True)

    def partially_finish_object(self):
        object_prob = self.current_object_prob
        if object_prob is None:
            return

        self.probs_history.append((object_prob, np.zeros_like(object_prob)))
        self.states.append(self.states[-1])

        self.clicker.reset_clicks()
        self.reset_predictor()
        self.reset_init_mask()
        self.update_image_callback()

    def finish_object(self):
        print(f"Finish object")
        with open('experiment_results.txt', 'a') as f:
            f.write(self.checkpoint)
            f.write(f"\nClicks: {self.clicks}\n")
            f.write(f"Prediction thresholds: {self.thresholds}\n")
            f.write(f"Dice: {self.dice}\nIoU: {self.iou}\n\n")
        if self.current_object_prob is None:
            return

        self._result_mask = self.result_mask
        self.object_count += 1
        self.reset_last_object()

    def reset_last_object(self, update_image=True):
        self.states = []
        self.probs_history = []
        self.clicker.reset_clicks()
        self.reset_predictor()
        self.reset_init_mask()
        if update_image:
            self.update_image_callback()
        self.dice = [0.0]
        self.iou = [0.0]
        self.thresholds = [0.0]
        self.clicks = [0]
        self.n_clicks = 0
        self.calculate_score(mask1=None, print_only=True)

    def reset_predictor(self, predictor_params=None):
        if predictor_params is not None:
            self.predictor_params = predictor_params
        self.predictor = get_predictor(self.net, device=self.device,
                                       **self.predictor_params)
        if self.image is not None:
            self.predictor.set_input_image(self.image)

    def reset_init_mask(self):
        self._init_mask = None
        self.clicker.click_indx_offset = 0

    def calculate_score(self, mask1, print_only=False):
        if not print_only:
            mask = mask1 > self.prob_thresh
            dice = get_dice(self._ground_truth_mask, mask)
            iou = get_iou(self._ground_truth_mask, mask)
            self.dice.append(dice)
            self.iou.append(iou)
            self.clicks.append(self.n_clicks)
            self.thresholds.append(self.prob_thresh)
        print(f"N clicks: {self.clicks[-1]}")
        print(f"Threshold: {self.thresholds[-1]}")
        print(f"Dice: {self.dice[-1]}\nIoU: {self.iou[-1]}\n\n")

    @property
    def current_object_prob(self):
        if self.probs_history:
            current_prob_total, current_prob_additive = self.probs_history[-1]
            return np.maximum(current_prob_total, current_prob_additive)
        else:
            return None

    @property
    def is_incomplete_mask(self):
        return len(self.probs_history) > 0

    @property
    def result_mask(self):
        result_mask = self._result_mask.copy()

        if self.probs_history:
            result_mask[self.current_object_prob > self.prob_thresh] = self.object_count + 1
        return result_mask

    def get_visualization(self, alpha_blend, click_radius, save_values):
        if self.image is None:
            return None

        # print(f"gt min max: {np.min(self._ground_truth_mask)}, {np.max(self._ground_truth_mask)}")

        results_mask_for_vis = self.result_mask
        mask_region = (results_mask_for_vis > 0).astype(np.uint8)

        # print(f"mask region min max: {np.min(mask_region)}, {np.max(mask_region)}")
        if self._ground_truth_mask is not None and save_values:
            self.calculate_score(mask_region)

        vis = draw_with_blend_and_clicks(self.image, mask=results_mask_for_vis, alpha=alpha_blend,
                                         clicks_list=self.clicker.clicks_list, radius=click_radius)
        if self.probs_history:
            total_mask = self.probs_history[-1][0] > self.prob_thresh
            results_mask_for_vis[np.logical_not(total_mask)] = 0
            vis = draw_with_blend_and_clicks(vis, mask=results_mask_for_vis, alpha=alpha_blend)

        return vis
