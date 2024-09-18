from typing import Optional

import torch
from mmdet.models.utils import multi_apply
from mmdet.utils import reduce_mean
from mmengine.model import BaseModel, bias_init_with_prob, xavier_init
from mmengine.registry import MODELS
from scipy.optimize import linear_sum_assignment
from torch import nn

from model.losses.loss import FocalCost, PtsL1Cost
from model.utils.grid_mask import GridMask


class MapTRV2(BaseModel):
    def __init__(self,
                 use_grid_mask=False,
                 embed_dims=256,
                 img_backbone=None,
                 img_neck=None,
                 pv2bev_encoder=None,
                 decoder=None,
                 cls_loss=None,
                 pts_loss=None,
                 dir_loss=None,
                 num_cams=6,
                 num_vec_len=20,
                 num_vec_one2one=50,
                 num_class=3,
                 bev_h=1,
                 bev_w=1,
                 pc_range=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        # 网络部分
        self.use_grid_mask = use_grid_mask
        if self.use_grid_mask:
            self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.img_backbone = MODELS.build(img_backbone)
        self.img_neck = MODELS.build(img_neck)
        self.pv2bev_encoder = MODELS.build(pv2bev_encoder)
        self.decoder = MODELS.build(decoder)
        # 损失部分
        self.cls_loss = MODELS.build(cls_loss)
        self.cls_loss_t = FocalCost(weight=cls_loss['loss_weight'])
        self.pts_loss = MODELS.build(pts_loss)
        self.pts_loss_t = PtsL1Cost(weight=pts_loss['loss_weight'])
        self.dir_loss = MODELS.build(dir_loss)
        # 辅助参数
        self.embed_dims = embed_dims
        self.num_vec_len = num_vec_len
        self.num_vec_one2one = num_vec_one2one
        self.num_cams = num_cams
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.pc_range = pc_range
        if self.cls_loss.use_sigmoid:
            self.cls_out_channels = num_class
        else:
            self.cls_out_channels = num_class + 1

        self._init_layers()

    def _init_layers(self):
        reg_branch = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, 2)
        )
        cls_branch = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.cls_out_channels)
        )
        self.reg_branches = nn.ModuleList([reg_branch for _ in range(self.decoder.num_layers)])
        self.cls_branches = nn.ModuleList([cls_branch for _ in range(self.decoder.num_layers)])
        self.reference_points = nn.Linear(self.embed_dims, 2)
        self.level_embeds = nn.Parameter(torch.Tensor(1, self.embed_dims))
        self.cams_embeds = nn.Parameter(torch.Tensor(self.num_cams, self.embed_dims))
        # embedding
        self.instance_embedding = nn.Embedding(self.num_vec_one2one, self.embed_dims * 2)
        self.pts_embedding = nn.Embedding(self.num_vec_len, self.embed_dims * 2)
        
        #channel mask
        self.channel_mlp=nn.Linear(self.bev_h*self.bev_w,3)

    def init_weights(self):
        if self.cls_loss.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)
        nn.init.normal_(self.level_embeds)
        nn.init.normal_(self.cams_embeds)
        xavier_init(self.reference_points.weight, distribution='uniform')
        super().init_weights()

    def extract_feat(self, img):
        """
        初步处理
        """
        b, n, c, h, w = img.size()
        img = img.reshape(b * n, c, h, w)
        if self.use_grid_mask:
            img = self.grid_mask(img)
        img_feats = self.img_backbone(img)#利用 backbone抽取图像特征
        img_feats = self.img_neck(img_feats)#经过neck加强特征
        _, c, h, w = img_feats[0].size()
        img_feats = img_feats[0].reshape(b, n, c, h, w)
        return img_feats

    def extract_out(self, img_feats: torch.Tensor, bev_embed: torch.Tensor, prev_bev=None, return_intermediate=True):
        b, num_cam, c, h, w = img_feats.shape
        d_type = bev_embed.dtype
        if self.training:
            num_vec = self.num_vec_one2one
        else:
            num_vec = self.num_vec_one2one
        pts_embeds = self.pts_embedding.weight.unsqueeze(0)#1.61
        instance_embeds = self.instance_embedding.weight[0:num_vec].unsqueeze(1)#0.01
        object_query_embed = (pts_embeds + instance_embeds).flatten(0, 1).to(d_type)
        query_pos, query = torch.split(object_query_embed, self.embed_dims, dim=1)#1200 256 
        query_pos = query_pos.unsqueeze(0).expand(b, -1, -1)#1 1200 256        60*20
        query = query.unsqueeze(0).expand(b, -1, -1)#1 1200 256
        
        
        
        
        
        
        #channel网络
        bs,num_query,num_channel=query_pos.shape
        query_pos=query_pos.reshape(bs*3,num_query//3,num_channel)#400为一类
        query=query.reshape(bs*3,num_query//3,num_channel)
        

        
        bs,num_grid,num_channel=bev_embed.shape
        bev_embed_input=self.channel_mlp(bev_embed.permute(0,2,1))#1 256 3
        bev_embed_input=bev_embed_input.sigmoid()
        bev_embed_input=bev_embed_input.permute(0,2,1)[:,:,None,:]#1 3 1 256
        
        bev_embed_input=bev_embed_input*bev_embed[:,None,:,:]#1 3 20000 256
        bev_embed_input=bev_embed_input.reshape(bs*3,num_grid,num_channel)
        bev_embed=bev_embed_input
        
        #channel 网络
        
        
        reference_points = self.reference_points(query_pos).sigmoid()

        query = query.permute(1, 0, 2)#前20个是1类，后20个是一类
        query_pos = query_pos.permute(1, 0, 2)
        bev_embed = bev_embed.permute(1, 0, 2)

        self_attn_mask = (
            torch.zeros([num_vec, num_vec]).bool().to(img_feats.device)
        )
        self_attn_mask[self.num_vec_one2one:, 0: self.num_vec_one2one, ] = True
        self_attn_mask[0: self.num_vec_one2one, self.num_vec_one2one:, ] = True

        
        inter_states, inter_references = self.decoder(#将每个query单独做一个批次
            query=query,
            key=None,
            value=bev_embed,
            query_pos=query_pos,
            reg_branches=self.reg_branches,
            reference_points=reference_points,
            spatial_shapes=torch.tensor([[self.bev_h, self.bev_w]], device=img_feats.device),
            level_start_index=torch.tensor([0], device=query.device),
            return_intermediate=return_intermediate,
            self_attn_mask=None,
            num_vec=num_vec,
            num_pts_per_vec=self.num_vec_len,
        )
        
        #channel 网络
        if return_intermediate:
            temp_inter_states=inter_states#6 400 3 256
            temp_inter_states=temp_inter_states.permute(0,2,1,3)#6 3 400 256
            num_layer,bs_m,num_query_m,num_channel=temp_inter_states.shape
            temp_inter_states=temp_inter_states.reshape(num_layer,bs_m//3,num_query_m*3,num_channel)
            temp_inter_states=temp_inter_states.permute(0,2,1,3)
            
            inter_states=temp_inter_states
            
            temp_inter_references=inter_references#6 3 400 2
            num_layer,bs_m,num_query_m,num_point=temp_inter_references.shape
            temp_inter_references=temp_inter_references.reshape(num_layer,bs_m//3,num_query_m*3,num_point)
            inter_references=temp_inter_references
        else:
            temp_inter_states=inter_states#400 3 256
            temp_inter_states=temp_inter_states.permute(1,0,2)#3 400 256
            bs_m,num_query_m,num_channel=temp_inter_states.shape
            temp_inter_states=temp_inter_states.reshape(bs_m//3,num_query_m*3,num_channel)
            temp_inter_states=temp_inter_states.permute(1,0,2)
            
            inter_states=temp_inter_states
            
            temp_inter_references=inter_references#3 400 2
            bs_m.num_query_m,num_point=temp_inter_references.shape
            temp_inter_references=temp_inter_references.reshape(bs_m//3,num_query_m*3,num_point)
            inter_references=temp_inter_references
        
        #channel 网络
        
        if return_intermediate:
            inter_states = inter_states.permute(0, 2, 1, 3)
        else:
            inter_states = inter_states.permute(1, 0, 2)
        return inter_states, inter_references

    def get_pred_dicts(self, inter_states, inter_references, one2many=False):
        outputs_classes_one2one = []
        outputs_pts_coords_one2one = []

        lvs, b, vec, dim = inter_states.shape
        for l in range(lvs):
            outputs_class = self.cls_branches[l](inter_states[l].reshape(b, -1, self.num_vec_len, dim).mean(2))
            outputs_pts_coord = self.transform_box(inter_references[l], num_vec=vec // self.num_vec_len)

            outputs_classes_one2one.append(outputs_class[:, 0:self.num_vec_one2one])#前50个为one，后面为one2many
            outputs_pts_coords_one2one.append(outputs_pts_coord[:, 0:self.num_vec_one2one])
        outputs_classes_one2one = torch.stack(outputs_classes_one2one)
        outputs_pts_coords_one2one = torch.stack(outputs_pts_coords_one2one)
        outs = {
            'one_cls_scores': outputs_classes_one2one,
            'one_pts_preds': outputs_pts_coords_one2one
        }
        return outs

    def transform_box(self, pts, num_vec=50, y_first=False):
        pts = pts.reshape(-1, num_vec, self.num_vec_len, 2)
        return pts

    def forward(self,
                inputs: torch.Tensor,#1 2 6 3 480 800
                data_samples: Optional[list] = None, #len 2
                mode: str = 'tensor'):
        if mode == 'loss':
            return self.forward_train(inputs, data_samples)
        elif mode == 'predict':
            return self.forward_test(inputs, data_samples)
        return None

    def forward_train(self, inputs: torch.Tensor, data_samples):
        inputs = inputs.permute(1, 0, 2, 3, 4, 5)#1 1 6 3 480 800
        len_q, bz, cam, c, h, w = inputs.shape
        if len_q == 2:
            prev_bev = None
        else:
            prev_bev = None
        img = inputs[-1, ...]
        img_metas = data_samples[-1]
        img_feats = self.extract_feat(img)#torch.float32->torch.float16
        # bev_feat b, dim, h，w
        # depth b, c, d, h, w
        bev_feat = self.pv2bev_encoder(img_feats, img_metas)#torch.float16->torch.float32
        inter_states, inter_references = self.extract_out(img_feats, bev_feat, prev_bev=prev_bev)#torch.float32 ->torch.float32 torch.float32
        # 计算loss
        loss = self.loss(inter_states,
                         inter_references,
                         img_metas)
        return loss

    def forward_test(self, inputs: torch.Tensor, data_samples):
        # TODO 实现历史帧
        img_feats = self.extract_feat(inputs)#8.86
        bev_feat = self.pv2bev_encoder(img_feats, data_samples)#9.83
        inter_states, inter_references = self.extract_out(img_feats, bev_feat, prev_bev=None, return_intermediate=False)
        cls_scores = self.cls_branches[-1](inter_states.reshape(1, -1, self.num_vec_len, self.embed_dims).mean(2))[0]
        pts_pred = self.transform_box(inter_references, num_vec=self.num_vec_one2one)[0]

        cls_scores = cls_scores.sigmoid()
        scores, index = cls_scores.view(-1).topk(50)
        labels = index % self.cls_out_channels
        pts_idx = index // self.cls_out_channels
        pts_pred = pts_pred[pts_idx]

        pts_pred[:, :, 0:1] = pts_pred[:, :, 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
        pts_pred[:, :, 1:2] = pts_pred[:, :, 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
        predictions_dict = {
            'scores': scores.cpu(),
            'labels': labels.cpu(),
            'pts': pts_pred.cpu(),
        }
        return [predictions_dict]

    def loss(self,
             inter_states,
             inter_references,
             img_metas) -> dict:#计算损失函数
        loss = dict()
        gt_label = []
        gt_pts = []
        for img_meta in img_metas:
            gt_label.append(img_meta.gt_label)
            gt_pts.append(img_meta.gt_pts)
        pred_dicts = self.get_pred_dicts(inter_states, inter_references, one2many=False)
        one_gt_label = [gt_label for _ in range(self.decoder.num_layers)]
        one_gt_pts = [gt_pts for _ in range(self.decoder.num_layers)]
        cls_loss, pts_loss, dir_loss = multi_apply(self.loss_single,
                                                   pred_dicts['one_cls_scores'], pred_dicts['one_pts_preds'],
                                                   one_gt_label, one_gt_pts)
        for i in range(len(cls_loss) - 1):
            loss[f'd{i}.cls_loss'] = cls_loss[i]
            loss[f'd{i}.pts_loss'] = pts_loss[i]
            loss[f'd{i}.dir_loss'] = dir_loss[i]
        loss['cls_loss'] = cls_loss[-1]
        loss['pts_loss'] = pts_loss[-1]
        loss['dir_loss'] = dir_loss[-1]
        return loss

    def loss_single(self, p_cls, p_pts, gt_cls, gt_pts):
        label, label_weights, pts_targets, pts_weights, pos_ind, neg_ind = multi_apply(self.get_target,
                                                                                       p_cls, p_pts,
                                                                                       gt_cls, gt_pts)
        num_total_pos = sum(pos_ind)
        num_total_neg = sum(neg_ind)
        label = torch.cat(label, 0)
        label_weights = torch.cat(label_weights, 0)
        pts_targets = torch.cat(pts_targets, 0)
        pts_weights = torch.cat(pts_weights, 0)

        p_cls = p_cls.reshape(-1, self.cls_out_channels)
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * 0
        cls_avg_factor = reduce_mean(p_cls.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)
        cls_loss = self.cls_loss(p_cls, label, label_weights, avg_factor=cls_avg_factor)

        num_total_pos = torch.clamp(reduce_mean(p_cls.new_tensor([num_total_pos])), min=1).item()
        p_pts = p_pts.reshape(-1, self.num_vec_len, 2)
        is_not_nan = torch.isfinite(pts_targets).all(dim=-1).all(dim=-1)
        pts_loss = self.pts_loss(p_pts[is_not_nan, ...],
                                 pts_targets[is_not_nan, ...],
                                 pts_weights[is_not_nan, ...],
                                 avg_factor=num_total_pos)

        dir_weights = pts_weights[:, :-1, 0]
        d_pts = p_pts[:, 1:, :] - p_pts[:, :-1, :]
        d_targe = pts_targets[:, 1:, :] - pts_targets[:, :-1, :]
        dir_loss = self.dir_loss(d_pts[is_not_nan, ...],
                                 d_targe[is_not_nan, ...],
                                 dir_weights[is_not_nan, ...],
                                 avg_factor=num_total_pos)

        return torch.nan_to_num(cls_loss), torch.nan_to_num(pts_loss), torch.nan_to_num(dir_loss)

    def get_target(self, p_cls, p_pts, gt_cls, gt_pts):
        num_gts = gt_cls.shape[0]
        num_pred = p_cls.shape[0]
        gt_pts = gt_pts.to(p_pts.device)
        gt_cls = gt_cls.to(p_cls.device)

        assigned_label = torch.full((num_pred,), self.cls_out_channels, dtype=torch.long, device=p_cls.device)
        label_weights = torch.ones_like(assigned_label)
        pts_targets = torch.zeros_like(p_pts)
        pts_weights = torch.zeros_like(p_pts)

        if num_gts == 0:
            return assigned_label, label_weights, pts_targets, pts_weights, num_gts, num_pred
        else:
            cls_cost = self.cls_loss_t(p_cls, gt_cls)

            gt_pts[..., 0:1] = (gt_pts[..., 0:1] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])#gts 归一化
            gt_pts[..., 1:2] = (gt_pts[..., 1:2] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])#gts 归一化
            pts_cost = self.pts_loss_t(p_pts, gt_pts).reshape(num_pred, num_gts, -1)
            pts_cost, order_index = torch.min(pts_cost, 2)

            cost = cls_cost + pts_cost
            # 匈牙利匹配，转到cpu上
            matched_row_inds, matched_col_inds = linear_sum_assignment(cost.detach().cpu())
            matched_row_inds = torch.from_numpy(matched_row_inds).to(p_cls.device)
            matched_col_inds = torch.from_numpy(matched_col_inds).to(p_cls.device)

            assigned_label[matched_row_inds] = gt_cls[matched_col_inds]

            pts_targets[matched_row_inds] = gt_pts[matched_col_inds, order_index[matched_row_inds, matched_col_inds]]
            pts_weights[matched_row_inds] = 1.0

            return assigned_label, label_weights, pts_targets, pts_weights, num_gts, num_pred - num_gts
