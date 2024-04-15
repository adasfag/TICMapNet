from mmengine import read_base

from datasets.fusion.openlane_map_dataset_fusion import CustomOpenlaneMapDatasetFusion
map_class = ('white-dash', 'white-solid',
             'yellow-dash', 'yellow-solid',
             'curbside')



from model.decoder.fusion.tiofusionva_decoder import FusionMapTRDecoder,TransformerDecoderLayer


from mmengine.config import read_base
from mmdet3d.datasets import LoadPointsFromFile

from datasets.nuscenes_map_dataset import CustomNuScenesMapDataset
from datasets.pipeline.formating import PackDataToInputs, MakeLineGts
from datasets.pipeline.loading import LoadMultiViewImageFromFiles
from datasets.pipeline.transform_3d import ScaleImageMultiViewImage, PhotoMetricDistortion3DMultiViewImage, \
    NormalizeMultiViewImage, PointToDepthMultiViewImage, PadMultiViewImageAndDepth
from mmcv.ops import MultiScaleDeformableAttention
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmdet.models.backbones.resnet import ResNet
from mmdet.models.necks.fpn import FPN
from torch.nn import ReLU, Dropout
from model.pv2bev_encoder.LSSTransform import LSSTransform
from model.evaluator.fusion.tiofusion_metric import Metric



from model.losses.loss import SimpleLoss, PtsL1Loss, PtsDirCosLoss
from mmdet.models import FocalLoss


from model.pv2bev_encoder.BEVFormEncoder import BEVFormerEncoder
from model.pv2bev_encoder.BEVFormEncoder import BEVFormerLayer
from model.pv2bev_encoder.temporal_self_attention import TemporalSelfAttention
from model.pv2bev_encoder.geometry_kernel_attention import GeometryKernelAttention
from model.pv2bev_encoder.geometry_kernel_attention import GeometrySptialCrossAttention


with read_base():
    from .default_runtime import *
from model.detectors.fusion.tiofusion import TIOFUSION

from mmengine.dataset import DefaultSampler


dim = 256
num_vec_len = 10
bev_h = 100
bev_w = 100



num_cams=1

pc_range = (-15.0, 0.0, -2.0, 15.0, 40.0, 2.0)
voxel_size = (0.15, 0.15, 4)
queue_len = 2










model_root = './'
data_root='./data/images/'

batch_size=8
bev_h = 100
bev_w = 100
num_vec_len = 10
num_vec_one2one=50

mean = (123.675, 116.28, 103.53)
std = (58.395, 57.12, 57.375)


bev_loss = False
pv_loss = False



model = dict(
    type=TIOFUSION,
    use_grid_mask=True,
    embed_dims=dim,
    num_cams=num_cams,
    num_vec_len=num_vec_len,
    num_vec_one2one=num_vec_one2one,
    bev_h=bev_h,
    bev_w=bev_w,
    num_class=len(map_class),
    pc_range=pc_range,
    loss_sim_weight=0.01,
    img_backbone=dict(
        type=ResNet,
        depth=50,
        out_indices=(3,),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='./ckpts/resnet50-11ad3fa6.pth'
        )
    ),
    img_neck=dict(
        type=FPN,
        in_channels=[2048],
        out_channels=dim,
        add_extra_convs='on_output',
        num_outs=1,
        relu_before_extra_convs=True
    ),
    
    pv2bev_encoder=dict(
                type=BEVFormerEncoder,
                num_layers=1,
                pc_range=pc_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                num_cams=num_cams,
                bev_h=bev_h,
                bev_w=bev_w,
                transformerlayers=dict(
                    type=BEVFormerLayer,
                    attn_cfgs=[
                        dict(
                            type=TemporalSelfAttention,
                            embed_dims=dim,
                            num_levels=1),
                        dict(
                            type=GeometrySptialCrossAttention,
                            pc_range=pc_range,
                            num_cams=num_cams,
                            attention=dict(
                                type=GeometryKernelAttention,
                                embed_dims=dim,
                                num_heads=4,
                                dilation=1,
                                kernel_size=(3,5),
                                num_levels=1),
                            embed_dims=dim,
                        )
                    ],
                    feedforward_channels=dim,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
     decoder=dict(
                type=FusionMapTRDecoder,
                num_layers=6,
                transformerlayers=dict(
                    type=TransformerDecoderLayer,
                    attn_cfgs=[
                        dict(
                            type=MultiheadAttention,
                            embed_dims=dim,
                            num_heads=8,
                            dropout=0.1),
                          dict(
                            type=MultiScaleDeformableAttention,
                            embed_dims=dim,
                            num_levels=1
                            )
                    ],

                    ffn_cfgs=dict(
                        type=FFN,
                     embed_dims=256,
                        feedforward_channels=dim * 2,
                        num_fcs=2,
                        ffn_drop=0.1,
                        act_cfg=dict(type=ReLU, inplace=True),
                        ),
                    feedforward_channels=dim,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
    cls_loss=dict(
        type=FocalLoss,
        use_sigmoid=True,
        gamma=2.0,
        alpha=0.25,
        loss_weight=2.0
    ),
    pts_loss=dict(
        type=PtsL1Loss,
        loss_weight=5.0
    ),
    dir_loss=dict(
        type=PtsDirCosLoss,
        loss_weight=0.005
    )
)


train_pipeline = [
    dict(type=LoadMultiViewImageFromFiles, to_float32=True),
    dict(type=ScaleImageMultiViewImage, scale=5/12),
    dict(type=PhotoMetricDistortion3DMultiViewImage),
    dict(type=NormalizeMultiViewImage, mean=mean, std=std),
    dict(type=PadMultiViewImageAndDepth, size_divisor=32),
    dict(type=MakeLineGts,
         num_vec_len=num_vec_len,
         bev=(bev_h, bev_w),
         bev_loss=bev_loss,
         pv_loss=pv_loss,
         bev_down_sample=1,
         feat_down_sample=32,
         z_min=3,
         pts_pattern='v1' 
         ),
    dict(type=PackDataToInputs)
]
test_pipeline = [
    dict(type=LoadMultiViewImageFromFiles, to_float32=True),
    dict(type=ScaleImageMultiViewImage, scale=0.5),
    dict(type=NormalizeMultiViewImage, mean=mean, std=std),
    dict(type=PadMultiViewImageAndDepth, size_divisor=32),
    dict(type=MakeLineGts,
         num_vec_len=num_vec_len,
         bev=(bev_h, bev_w),
         bev_loss=False,
         pv_loss=False,
         pts_pattern='v1'  
         ),
    dict(type=PackDataToInputs)
]


train_dataloader = dict(
    dataset=dict(
        type=CustomOpenlaneMapDatasetFusion,
        ann_file=model_root+'openlane_map_train.pkl',
        data_path=data_root,
        pipeline=train_pipeline,
        queue_length=queue_len,
    ),
    sampler=dict(
        type='DefaultSampler',
        shuffle=True),
    collate_fn=dict(type='default_collate'),
    batch_size=batch_size,
    pin_memory=True,
    num_workers=4)


val_dataloader = dict(
    dataset=dict(
        type=CustomOpenlaneMapDatasetFusion,
        ann_file=model_root+'openlane_map_val.pkl',
        data_path=data_root,
        pipeline=test_pipeline,
        queue_length=queue_len,
        test_mode=True
    ),
    sampler=dict(
        type='DefaultSampler',
        shuffle=False),
    collate_fn=dict(type='default_collate'),
    batch_size=1,
    pin_memory=True,
    num_workers=2)
test_dataloader = val_dataloader

train_cfg = dict(
    by_epoch=True,
    max_epochs=24,
    val_begin=100,
    val_interval=100
)
val_cfg = dict(type='ValLoop')
val_evaluator = dict(
    type=Metric,
    metric='chamfer',
    classes=map_class,
    score_thresh=0.00,
    prefix=''
)
test_cfg = dict(type='TestLoop')
test_evaluator = dict(
    type=Metric,
    metric='chamfer',
    classes=map_class,
    save=True,
    save_path='',
    score_thresh=0.00,
    prefix=''
)

optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=1.5e-4,
        weight_decay=0.00125
    ),
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }
    ),
    clip_grad=dict(max_norm=35, norm_type=2),
    
)

end = 4000 

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1.0 / 3, by_epoch=False, begin=0, end=end),
    dict(
        type='CosineAnnealingLR',
        T_max=train_cfg['max_epochs'],
        by_epoch=True,
        convert_to_iter_based=True,
        eta_min_ratio=1e-3
    )
]

randomness = dict(seed=0)

load_from = None
resume = False

default_hooks['checkpoint'] = dict(type='CheckpointHook', interval=1)