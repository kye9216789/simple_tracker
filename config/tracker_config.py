cfg = dict(
    tracker=dict(
        batch_size=5,
    ),
    bbox=dict(
        repository_dir='/home/ye/github/mp_mmdetection/',
        work_dir='work_dirs/190604_retinanet_x101_32x4d_fpn_1x'
    ),
    lstm=dict(
        num_layers=2,
        hidden_dim=500,
        out_dim=2,
        seq_len=3
    ),
    train=dict(
        epoch=50,
        validation_interval=1,
        
    )
)