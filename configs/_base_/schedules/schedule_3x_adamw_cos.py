# training schedule for 2x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=36, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# optimizer
# for batch in each gpu is 128, 8 gpu
# lr = 5e-4 * 128 * 8 / 512 = 0.001
lr = 5e-4 * 128 * 8 / 512
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=lr,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999))
)

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=True,
        begin=0,
        end=1,
        # update by iter
        convert_to_iter_based=True,
    ),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        T_max=35,
        eta_min=lr * 0.01,
        by_epoch=True,
        begin=1,
        end=36,
        # update by iter
        convert_to_iter_based=True,
    )
]


# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
