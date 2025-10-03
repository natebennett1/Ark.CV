# Minimal MMDetection config for single-class Cascade R-CNN R50-FPN
_base_ = [
    r'C:\Users\20ben\Ark.CV\mmdetection\configs\_base_\models\cascade-rcnn_r50_fpn.py',
    r'C:\Users\20ben\Ark.CV\mmdetection\configs\_base_\datasets\coco_detection.py',
    r'C:\Users\20ben\Ark.CV\mmdetection\configs\_base_\schedules\schedule_1x.py',
    r'C:\Users\20ben\Ark.CV\mmdetection\configs\_base_\default_runtime.py',
]


metainfo = dict(classes=("fish",))

# dataset
data_root = "./mmdet_singleclass/"
train_ann = "train/_annotations.coco.json"
val_ann   = "valid/_annotations.coco.json"
test_ann  = "test/_annotations.coco.json"

train_dataloader = dict(
    dataset=dict(
        type="CocoDataset",
        data_root=data_root,
        ann_file=train_ann,
        data_prefix=dict(img="train/"),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=True),
    )
)
val_dataloader = dict(
    dataset=dict(
        type="CocoDataset",
        data_root=data_root,
        ann_file=val_ann,
        data_prefix=dict(img="valid/"),
        metainfo=metainfo,
    )
)
test_dataloader = dict(
    dataset=dict(
        type="CocoDataset",
        data_root=data_root,
        ann_file=test_ann,
        data_prefix=dict(img="test/"),
        metainfo=metainfo,
    )
)

# model: set num_classes=1 for each cascade head
model = dict(
    roi_head=dict(
        bbox_head=[
            dict(type="Shared2FCBBoxHead", num_classes=1),
            dict(type="Shared2FCBBoxHead", num_classes=1),
            dict(type="Shared2FCBBoxHead", num_classes=1),
        ]
    )
)

# training tricks (optional—tune later)
train_cfg = dict(rcnn=dict(sampler=dict(num=512, pos_fraction=0.25)))
optim_wrapper = dict(optimizer=dict(lr=0.02))  # scale by batch/GPU if needed
work_dir = "./work_dirs/arkcv_cascade_r50_fpn_fish"
