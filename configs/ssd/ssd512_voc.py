input_size = 512
model = dict(
    neck=dict(
        out_channels=(512, 1024, 512, 256, 256, 256, 256),
        level_strides=(2, 2, 2, 2, 1),
        level_paddings=(1, 1, 1, 1, 1),
        last_kernel_size=4),
    bbox_head=dict(
        in_channels=(512, 1024, 512, 256, 256, 256, 256),
        anchor_generator=dict(
            type='SSDAnchorGenerator',
            scale_major=False,
            input_size=input_size,
            basesize_ratio_range=(0.1, 0.9),
            strides=[8, 16, 32, 64, 128, 256, 512],
            ratios=[[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]])))
# dataset settings

checkpoint_config = dict(interval=3)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
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
uncertainty_pool2 = 'objectSum_scaleAvg_classSum'
# 'scaleAvg_classAvg', 'scaleSum_classSum', scaleAvg_classSum', 'scaleSum_classAvg'

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
        # ann_file=data_root + 'VOC2007/ImageSets/Main/mini_test.txt',
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007/',
        pipeline=test_pipeline),
    test=dict(
        # type=dataset_type,
        # ann_file=data_root + 'VOC2007/ImageSets/Main/mini_test.txt',
        # img_prefix=data_root + 'VOC2007/',
        # pipeline=train_pipeline))
        type=dataset_type,
        ann_file=[
            data_root + 'VOC2007/ImageSets/Main/trainval.txt',
            data_root + 'VOC2012/ImageSets/Main/trainval.txt',
        ],
        img_prefix=[data_root + 'VOC2007/', data_root + 'VOC2012/'],
        pipeline=train_pipeline))


# evaluation = dict(interval=3, metric='mAP', show=True, isUnc=False, out_dir='show_dir')
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

