checkpoint_config = dict(interval=3)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
custom_hooLoks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
uncertainty_pool = 'Entropy_NMS'
# 'Random', 'Entropy_ALL', 'Entropy_NMS', Entropy_NoNMS'
uncertainty_type = 'Epistemic'
# 'Shannon', 'Aleatoric', 'Epistemic', 'Total'
uncertainty_pool2 = 'objectSum_scaleMax_classSum'
# 'scaleAvg_classAvg', 'scaleSum_classSum', scaleAvg_classSum', 'scaleSum_classAvg'

# model settings
input_size = 300
model = dict(
    type='SSD_L_SingleStageDetector',
    backbone=dict(
        type='SSDVGG',
        depth=16,
        with_last_pool=False,
        ceil_mode=True,
        out_indices=(3, 4),
        out_feature_indices=(22, 34),
        init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://vgg16_caffe')),
    neck=dict(
        type='SSDNeck',
        in_channels=(512, 1024),
        out_channels=(512, 1024, 512, 256, 256, 256),
        level_strides=(2, 2, 1, 1),
        level_paddings=(1, 1, 0, 0),
        l2_norm_scale=20),
    bbox_head=dict(
        type='MyLSSDHead',
        num_classes=20,
        in_channels=(512, 1024, 512, 256, 256, 256),
        anchor_generator=dict(
            type='SSDAnchorGenerator',
            scale_major=False,
            input_size=input_size,
            basesize_ratio_range=(0.15, 0.9),
            strides=[8, 16, 32, 64, 100, 300],
            ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2])),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.,
            ignore_iof_thr=-1,
            gt_max_assign_all=False),
        smoothl1_beta=1.,
        allowed_border=-1,
        pos_weight=-1,
        neg_pos_ratio=3,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        nms=dict(type='nms', iou_threshold=0.5),
        min_bbox_size=0,
        score_thr=0.02,
        max_per_img=200,
        uncertainty_pool = uncertainty_pool)
)
# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(policy='step', step=[2])
runner = dict(type='MyEpochBasedRunnerLSSD', max_epochs=3)

# dataset settings
dataset_type = 'VOCDataset'
data_root = '/drive1/YH/datasets/VOCdevkit/VOCdevkit/'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 4)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=(300, 300), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(300, 300),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=0,
    train=dict(
        type='RepeatDataset', times=1, dataset=dict(
            type=dataset_type,
            ann_file=[
                data_root + 'VOC2007/ImageSets/Main/trainval.txt',
                data_root + 'VOC2012/ImageSets/Main/trainval.txt'
            ],
            img_prefix=[data_root + 'VOC2007/', data_root + 'VOC2012/'],
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=[
            data_root + 'VOC2007/ImageSets/Main/trainval.txt',
            data_root + 'VOC2012/ImageSets/Main/trainval.txt',
        ],
        img_prefix=[data_root + 'VOC2007/', data_root + 'VOC2012/'],
        pipeline=train_pipeline))

evaluation = dict(interval=5, metric='mAP', show=False, isUnc=False, out_dir=None)
# The moment when the learning rate drops can be changed here.
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[1])
k = 10000
X_S_size = 1000
X_L_0_size = 1000
# The active learning cycles can be changed here.
cycles = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
epoch_ratio = [5, 1]
outer_epoch = 2
# The repeat time for the labeled sets and unlabeled sets can be changed here.
# The number of repeat times can be equivalent to the number of actual training epochs.
X_L_repeat = 16
X_U_repeat = 16
# The hyper-parameters lambda and k can be changed here.
train_cfg = dict(param_lambda = 0.5)
