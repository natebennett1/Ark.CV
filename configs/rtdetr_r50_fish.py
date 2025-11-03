"""RT-DETR R50 configuration tuned for single-class fish detection."""
_base_ = "mmdet::rtdetr/rtdetr_r50_8xb2-12e_coco.py"

default_scope = "mmdet"

metainfo = dict(classes=("Fish",), palette=[(0, 255, 0)])

data_root = "./"
img_dir = "images/"

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="RandomChoiceResize",
        scales=[(1280, 960), (1280, 1080), (1280, 1280)],
        keep_ratio=True,
    ),
    dict(type="RandomFlip", prob=0.5),
    dict(
        type="PhotoMetricDistortion",
        brightness_delta=16,
        contrast_range=(0.85, 1.15),
        saturation_range=(0.85, 1.15),
        hue_delta=5,
    ),
    dict(type="PackDetInputs"),
]

val_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(1280, 960), keep_ratio=True),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="PackDetInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(1280, 960), keep_ratio=True),
    dict(type="PackDetInputs"),
]

train_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file="anns_det_fish_only_train.json",
        data_prefix=dict(img=img_dir),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline,
    )
)
val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file="anns_det_fish_only_val.json",
        data_prefix=dict(img=img_dir),
        pipeline=val_pipeline,
    )
)
test_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file="anns_det_fish_only_val.json",
        data_prefix=dict(img=img_dir),
        pipeline=test_pipeline,
    )
)

model = dict(bbox_head=dict(num_classes=1))

# Slightly lower base LR to account for smaller batch sizes when training on a single GPU.
optim_wrapper = dict(optimizer=dict(lr=2.5e-4))

val_evaluator = dict(ann_file=f"{data_root}anns_det_fish_only_val.json")
test_evaluator = dict(ann_file=f"{data_root}anns_det_fish_only_val.json")
