class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/data/wuyingjie/dzd/fourth_work/BAT2/BAT'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/data/wuyingjie/dzd/fourth_work/BAT2/BAT/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/data/wuyingjie/dzd/fourth_work/BAT2/BAT/pretrained_networks'
        self.got10k_val_dir = '/data/wuyingjie/dzd/fourth_work/BAT2/BAT/data/got10k/val'
        self.lasot_lmdb_dir = '/data/wuyingjie/dzd/fourth_work/BAT2/BAT/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/data/wuyingjie/dzd/fourth_work/BAT2/BAT/data/got10k_lmdb'
        self.trackingnet_lmdb_dir = '/data/wuyingjie/dzd/fourth_work/BAT2/BAT/data/trackingnet_lmdb'
        self.coco_lmdb_dir = '/data/wuyingjie/dzd/fourth_work/BAT2/BAT/data/coco_lmdb'
        self.coco_dir = '/data/wuyingjie/dzd/fourth_work/BAT2/BAT/data/coco'
        self.lasot_dir = '/data/wuyingjie/dzd/fourth_work/BAT2/BAT/data/lasot'
        self.got10k_dir = '/data/wuyingjie/dzd/fourth_work/BAT2/BAT/data/got10k/train'
        self.trackingnet_dir = '/data/wuyingjie/dzd/fourth_work/BAT2/BAT/data/trackingnet'
        self.depthtrack_dir = '/data/wuyingjie/dzd/fourth_work/BAT2/BAT/data/depthtrack/train'
        self.lasher_dir = '/data/wuyingjie/dzd/fourth_work/BAT2/BAT/data/lasher'
        self.visevent_dir = '/data/wuyingjie/dzd/fourth_work/BAT2/BAT/data/visevent/train'
        self.drgbt603_dir = '/data/wuyingjie/datasets'  
