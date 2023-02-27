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
model = dict(
    type='SSL_L_RetinaNet',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='Lambda_L2Net',
        num_classes=20,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='EDL_Softmax_FocalLoss',
            last_activation = 'relu',
            num_classes=20,
            annealing_step=10,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        neg_pos_ratio=0,
        bias = 'uniform',
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100,
        uncertainty_pool = uncertainty_pool)
)

# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(policy='step', step=[2])
runner = dict(type='MyEpochBasedRunnerLambda', max_epochs=3)

# dataset settings
dataset_type = 'VOCDataset'
data_root = '/drive1/YH/datasets/VOCdevkit/VOCdevkit/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1000, 600), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1000, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
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

evaluation = dict(interval=3, metric='mAP', show=False, isUnc=False, out_dir=None)
# The size of the initial labeled set and the newly selected sets after each cycle can be set here.
# Note that there are 16551 images in the PASCAL VOC 2007+2012 trainval sets.
X_S_size = 16551//40
X_L_0_size = 16551//20
# The active learning cycles can be changed here.
cycles = [0, 1, 2, 3, 4, 5, 6]
epoch_ratio = [3, 1]
outer_epoch = 2
# The repeat time for the labeled sets and unlabeled sets can be changed here.
# The number of repeat times can be equivalent to the number of actual training epochs.
X_L_repeat = 2
X_U_repeat = 2
# The hyper-parameters lambda and k can be changed here.
train_cfg = dict(param_lambda = 0.5)

k = 10000