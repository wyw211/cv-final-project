# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import pdb

import torch
from torch import nn
import torch.nn.functional as F

from fastreid.config import configurable
from fastreid.modeling.backbones import build_backbone
from fastreid.modeling.heads import build_heads
from fastreid.modeling.losses import *
from .build import META_ARCH_REGISTRY


class ViewPrototypeMemory(nn.Module):
    """Momentum prototypes per (identity, view) for the view-aware prototype loss (P1).

    For a feature f of identity y seen from view v, the loss is
        w_c * CE(cos(f, P[:, 1-v]) / t, y) + (1 - w_c) * CE(cos(f, P[:, v]) / t, y)
    over all identities whose prototype for that view exists. After the loss, each
    (y, v) prototype present in the batch moves towards the batch mean:
        P = normalize(m * P + (1 - m) * normalize(mean f)).
    Training only: inference does not use it.
    """

    def __init__(self, num_classes, dim, temp=0.05, momentum=0.2, cross_weight=0.5):
        super().__init__()
        self.temp, self.momentum, self.cross_weight = temp, momentum, cross_weight
        self.register_buffer('protos', torch.zeros(num_classes, 2, dim))
        self.register_buffer('valid', torch.zeros(num_classes, 2, dtype=torch.bool))

    def _term(self, f, targets, view):
        P = self.protos[:, view].clone()             # [C, D]; clone: updated in place later
        ok = self.valid[targets, view]              # samples whose positive exists
        if not ok.any():
            return f.sum() * 0.0
        logits = f @ P.t() / self.temp
        logits = logits.masked_fill(~self.valid[:, view].unsqueeze(0), float('-inf'))
        return F.cross_entropy(logits[ok], targets[ok])

    def forward(self, feats, targets, view_id):
        f = F.normalize(feats.float(), dim=-1)
        loss = 0.0
        for v in (0, 1):
            m = view_id == v
            if not m.any():
                continue
            loss = loss + m.float().mean() * (
                self.cross_weight * self._term(f[m], targets[m], 1 - v)
                + (1 - self.cross_weight) * self._term(f[m], targets[m], v))
        self._update(f.detach(), targets, view_id)
        return loss

    @torch.no_grad()
    def _update(self, f, targets, view_id):
        key = targets * 2 + view_id
        uniq, inv = key.unique(return_inverse=True)
        mean = torch.zeros(len(uniq), f.shape[1], device=f.device).index_add_(0, inv, f)
        mean = F.normalize(mean, dim=-1)
        ids, vs = uniq // 2, uniq % 2
        old = self.protos[ids, vs]
        fresh = ~self.valid[ids, vs]
        new = torch.where(fresh.unsqueeze(1), mean, self.momentum * old + (1 - self.momentum) * mean)
        self.protos[ids, vs] = F.normalize(new, dim=-1)
        self.valid[ids, vs] = True


@META_ARCH_REGISTRY.register()
class Baseline_multiview_ViSA(nn.Module):
    """
    Baseline architecture. Any models that contains the following two components:
    1. Per-image feature extraction (aka backbone)
    2. Per-image feature aggregation and loss computation
    """

    @configurable
    def __init__(
            self,
            *,
            backbone,
            heads,
            view_heads,
            local_heads,
            pixel_mean,
            pixel_std,
            use_visa=True,
            loss_kwargs=None,
            pcl=None,
            illum=None,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            backbone:
            heads:
            pixel_mean:
            pixel_std:
        """
        super().__init__()
        # backbone
        self.backbone = backbone

        # head
        self.heads = heads
        self.view_heads = view_heads
        self.local_heads = local_heads
        self.loss_kwargs = loss_kwargs

        self.use_visa = use_visa
        # P1: optional view-aware prototype memories (invariant global, local branch).
        self.pcl_weight = 0.0
        if pcl is not None and pcl['enabled']:
            self.pcl_weight = pcl['weight']
            mk = lambda: ViewPrototypeMemory(pcl['num_classes'], pcl['dim'], pcl['temp'],
                                             pcl['momentum'], pcl['cross_weight'])
            self.pcl_global, self.pcl_local = mk(), mk()

        # L3: illumination-consistency loss (training only).
        self.illum_weight = illum['weight'] if illum is not None and illum['enabled'] else 0.0
        self.illum_num = illum['num'] if illum is not None else 0

        self.register_buffer('pixel_mean', torch.Tensor(pixel_mean).view(1, -1, 1, 1), False)
        self.register_buffer('pixel_std', torch.Tensor(pixel_std).view(1, -1, 1, 1), False)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        heads = build_heads(cfg)
        
        cfg0 = cfg.clone()
        if cfg0.is_frozen(): cfg0.defrost()
        cfg0.MODEL.HEADS.NUM_CLASSES = 2
        view_heads = build_heads(cfg0)
        cfg0 = cfg.clone()

        local_heads = build_heads(cfg)

        return {
            'backbone': backbone,
            'heads': heads,
            'view_heads': view_heads,
            'local_heads': local_heads,
            'pixel_mean': cfg.MODEL.PIXEL_MEAN,
            'pixel_std': cfg.MODEL.PIXEL_STD,
            'use_visa': cfg.MODEL.USE_VISA,
            'pcl': {
                'enabled': cfg.MODEL.LOSSES.PCL.ENABLED,
                'weight': cfg.MODEL.LOSSES.PCL.WEIGHT,
                'temp': cfg.MODEL.LOSSES.PCL.TEMP,
                'momentum': cfg.MODEL.LOSSES.PCL.MOMENTUM,
                'cross_weight': cfg.MODEL.LOSSES.PCL.CROSS_WEIGHT,
                'num_classes': cfg.MODEL.HEADS.NUM_CLASSES,
                'dim': cfg.MODEL.BACKBONE.FEAT_DIM,
            },
            'illum': {
                'enabled': cfg.MODEL.LOSSES.ILLUM.ENABLED,
                'weight': cfg.MODEL.LOSSES.ILLUM.WEIGHT,
                'num': cfg.MODEL.LOSSES.ILLUM.NUM,
            },
            'loss_kwargs':
                {
                    # loss name
                    'loss_names': cfg.MODEL.LOSSES.NAME,

                    # loss hyperparameters
                    'ce': {
                        'eps': cfg.MODEL.LOSSES.CE.EPSILON,
                        'alpha': cfg.MODEL.LOSSES.CE.ALPHA,
                        'scale': cfg.MODEL.LOSSES.CE.SCALE,
                        'view_id': cfg.MODEL.LOSSES.CE.VIEW_ID,
                        'view_oreg': cfg.MODEL.LOSSES.CE.VIEW_OREG,
                        'view_lambda': cfg.MODEL.LOSSES.CE.VIEW_LAMBDA,
                    },
                    'tri': {
                        'margin': cfg.MODEL.LOSSES.TRI.MARGIN,
                        'norm_feat': cfg.MODEL.LOSSES.TRI.NORM_FEAT,
                        'hard_mining': cfg.MODEL.LOSSES.TRI.HARD_MINING,
                        'scale': cfg.MODEL.LOSSES.TRI.SCALE
                    },
                    'circle': {
                        'margin': cfg.MODEL.LOSSES.CIRCLE.MARGIN,
                        'gamma': cfg.MODEL.LOSSES.CIRCLE.GAMMA,
                        'scale': cfg.MODEL.LOSSES.CIRCLE.SCALE
                    },
                    'cosface': {
                        'margin': cfg.MODEL.LOSSES.COSFACE.MARGIN,
                        'gamma': cfg.MODEL.LOSSES.COSFACE.GAMMA,
                        'scale': cfg.MODEL.LOSSES.COSFACE.SCALE
                    },

                }
        }

    @property
    def device(self):
        return self.pixel_mean.device

    @staticmethod
    def _relight(raw):
        """Re-light raw [0, 255] images on the GPU: bright ones are darkened (gamma 2-4,
        contrast 0.3-0.7, desaturation 0.3-0.8, noise), dark ones brightened (gamma 0.4-0.75). Mirrors L1."""
        x = (raw / 255.).clamp(0, 1)
        n = x.shape[0]
        lum = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).mean(dim=(1, 2))
        dark = lum < 0.30
        u = torch.rand(n, device=x.device)
        g = torch.where(dark, 0.4 + 0.35 * u, 2.0 + 2.0 * u).view(n, 1, 1, 1)
        x = x.clamp_min(1e-6) ** g
        c = torch.where(dark, torch.ones_like(u), 0.3 + 0.4 * torch.rand(n, device=x.device)).view(n, 1, 1, 1)
        mu = x.mean(dim=(1, 2, 3), keepdim=True)
        x = mu + c * (x - mu)
        gray = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
        ks = torch.where(dark, torch.ones_like(u), 0.3 + 0.5 * torch.rand(n, device=x.device)).view(n, 1, 1, 1)
        x = ks * x + (1 - ks) * gray
        sigma = torch.where(dark, torch.zeros_like(u), 0.02 * torch.rand(n, device=x.device)).view(n, 1, 1, 1)
        x = x + sigma * torch.randn_like(x)
        return x.clamp(0, 1) * 255.

    def forward(self, batched_inputs):
        relit = None
        if self.training and self.illum_weight > 0:
            k = min(self.illum_num, batched_inputs['images'].shape[0])
            relit = self._relight(batched_inputs['images'][:k].clone())
        images = self.preprocess_image(batched_inputs)
        # print(batched_inputs.keys())
        if 'camids' in batched_inputs.keys():
            camids = batched_inputs['camids']
        else:
            camids = 0
        if 'viewids' in batched_inputs.keys():
            view = batched_inputs['viewids']
            view1_index = [index for index, content in enumerate(view) if content == 'Aerial']
            view2_index = [index for index, content in enumerate(view) if content == 'Ground']
            view_id = torch.tensor([0 if content == 'Aerial' else 1 for content in view]).to(device=camids.device)

        global_feats, local_feats, view_feats, prompt, load_balance_loss_s = self.backbone(images, camids, view_id)
        
        features = global_feats - view_feats
        # features = query_feats
        # features = global_feats
        if self.training:
            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            targets = batched_inputs["targets"]

            temp = torch.zeros((targets.shape[0])).long().to(targets.device)
            temp[view1_index] = 1
            targets_view = temp

            # PreciseBN flag, When do preciseBN on different dataset, the number of classes in new dataset
            # may be larger than that in the original dataset, so the circle/arcface will
            # throw an error. We just set all the targets to 0 to avoid this problem.
            if targets.sum() < 0: targets.zero_()

            outputs = self.heads(features, targets)
            outputs_global = self.heads(global_feats, targets)
            
            local_outputs = self.local_heads(local_feats, targets)
            view_outputs = self.view_heads(view_feats, targets_view)

            losses = self.losses(outputs, outputs_global, local_outputs, view_outputs, targets, targets_view)
            # Official value 0.0001 (hard-coded upstream); now MODEL.BACKBONE.MOE_BALANCE_LAMBDA.
            losses['load_balance_loss'] = getattr(self.backbone, 'moe_balance_lambda', 0.0001) * load_balance_loss_s
            if relit is not None:
                k = relit.shape[0]
                relit = (relit - self.pixel_mean) / self.pixel_std
                cam_k = camids[:k] if torch.is_tensor(camids) else camids
                g2, _, v2, _, _ = self.backbone(relit, cam_k, view_id[:k])
                f2 = torch.nn.functional.normalize((g2 - v2).float(), dim=1)
                f1 = torch.nn.functional.normalize(features[:k].detach().float(), dim=1)
                losses['loss_illum'] = self.illum_weight * (1 - (f1 * f2).sum(dim=1)).mean()
            if self.pcl_weight > 0:
                losses['loss_pcl'] = self.pcl_weight * self.pcl_global(outputs['features'], targets, view_id)
                if self.use_visa:
                    losses['loss_pcl_local'] = self.pcl_weight * self.pcl_local(local_outputs['features'], targets, view_id)
            return losses
        else:
            outputs = self.heads(features)
            local_outputs = self.local_heads(local_feats)
            if self.use_visa:
                outputs = torch.cat((outputs, local_outputs), dim=1)
            return outputs

    def preprocess_image(self, batched_inputs):
        """
        Normalize and batch the input images.
        """
        if isinstance(batched_inputs, dict):
            images = batched_inputs['images']
        elif isinstance(batched_inputs, torch.Tensor):
            images = batched_inputs
        else:
            raise TypeError("batched_inputs must be dict or torch.Tensor, but get {}".format(type(batched_inputs)))

        images.sub_(self.pixel_mean).div_(self.pixel_std)
        return images

    def losses(self, outputs, outputs_global, outputs_local, outputs_view, gt_labels, view_labels):
        """
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        # model predictions
        # fmt: off
        pred_class_logits = outputs['pred_class_logits'].detach()
        cls_outputs = outputs['cls_outputs']
        pred_features = outputs['features']
        
        view_pred_class_logits = outputs_view['pred_class_logits'].detach()
        view_cls_outputs = outputs_view['cls_outputs']
        view_pred_features = outputs_view['features']
        
        global_pred_class_logits = outputs_global['pred_class_logits'].detach()
        global_cls_outputs = outputs_global['cls_outputs']
        global_pred_features = outputs_global['features']

        local_pred_class_logits = outputs_local['pred_class_logits'].detach()
        local_cls_outputs = outputs_local['cls_outputs']
        local_pred_features = outputs_local['features']
        # fmt: on

        # Log prediction accuracy
        log_accuracy(pred_class_logits, gt_labels)

        loss_dict = {}
        loss_names = self.loss_kwargs['loss_names']

        view_kwargs = self.loss_kwargs.get('ce')
        view_id_flag = view_kwargs.get('view_id')
        view_oreg_flag = view_kwargs.get('view_oreg')
        view_lambda = view_kwargs.get('view_lambda')

        if 'CrossEntropyLoss' in loss_names:
            ce_kwargs = self.loss_kwargs.get('ce')
            loss_dict['loss_cls_id'] = cross_entropy_loss(
                cls_outputs,
                gt_labels,
                ce_kwargs.get('eps'),
                ce_kwargs.get('alpha')
            ) * ce_kwargs.get('scale')
            if self.use_visa:
                loss_dict['loss_cls_local'] = cross_entropy_loss(
                    local_cls_outputs,
                    gt_labels,
                    ce_kwargs.get('eps'),
                    ce_kwargs.get('alpha')
                ) * ce_kwargs.get('scale')

            if view_id_flag:
                loss_dict['loss_cls_view'] = cross_entropy_loss(
                    view_cls_outputs,
                    view_labels,
                    ce_kwargs.get('eps'),
                    ce_kwargs.get('alpha')
                ) * ce_kwargs.get('scale') * view_lambda

        if 'TripletLoss' in loss_names:
            tri_kwargs = self.loss_kwargs.get('tri')
            loss_dict['loss_triplet_id'] = triplet_loss(
                pred_features,
                gt_labels,
                tri_kwargs.get('margin'),
                tri_kwargs.get('norm_feat'),
                tri_kwargs.get('hard_mining')
            ) * tri_kwargs.get('scale')
            if self.use_visa:
                loss_dict['loss_triplet_local'] = triplet_loss(
                    local_pred_features,
                    gt_labels,
                    tri_kwargs.get('margin'),
                    tri_kwargs.get('norm_feat'),
                    tri_kwargs.get('hard_mining')
                ) * tri_kwargs.get('scale')

        # calc oreg loss part
        if view_oreg_flag:
            loss_dict['loss_oreg'] = torch.cosine_similarity(pred_features, view_pred_features).abs().mean() * view_lambda

        if 'CircleLoss' in loss_names:
            circle_kwargs = self.loss_kwargs.get('circle')
            loss_dict['loss_circle'] = pairwise_circleloss(
                pred_features,
                gt_labels,
                circle_kwargs.get('margin'),
                circle_kwargs.get('gamma')
            ) * circle_kwargs.get('scale')

        if 'Cosface' in loss_names:
            cosface_kwargs = self.loss_kwargs.get('cosface')
            loss_dict['loss_cosface'] = pairwise_cosface(
                pred_features,
                gt_labels,
                cosface_kwargs.get('margin'),
                cosface_kwargs.get('gamma'),
            ) * cosface_kwargs.get('scale')

        # prompt_ma = prompt.mean(dim=0)
        # gram_matrix = torch.mm(prompt_ma, prompt_ma.t())
        # n = gram_matrix.size(0)
        # off_diagonal = gram_matrix[~torch.eye(n, dtype=bool)]
        # loss_dict['prompt_loss'] = torch.sum(off_diagonal ** 2)

        return loss_dict
