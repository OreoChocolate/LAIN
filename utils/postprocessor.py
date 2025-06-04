import torch
import torch.nn as nn
import torch.nn.functional as F
import detr.util.box_ops as box_ops


# for DETR
class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes, return_score=False):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]
        if return_score:
            # new_score =  prob[..., :-1]
            new_score = out_logits
            results = [{'scores_a':s_a, 'scores': s, 'labels': l, 'boxes': b} for s_a, s, l, b in zip(new_score, scores, labels, boxes)]
        else:
            results = [{'scores': s, 'labels': l, 'boxes': b, 'feats': f} for s, l, b, f in zip(scores, labels, boxes,outputs['feats'])]

        return results