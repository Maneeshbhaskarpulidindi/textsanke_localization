from easydict import EasyDict
import torch

config = EasyDict()

# dataloader jobs number
config.num_workers = 4

# batch_size
config.batch_size = 8

# training epoch number
config.max_epoch = 600

config.start_epoch = 0

# learning rate
config.lr = 1e-4

# using GPU
config.cuda = True

config.n_disk =  15 #15

config.output_dir = 'newlogs/output'

config.save_dir = 'newlogs/save'

config.log_dir = 'newlogs/logs'

config.vis_dir = 'newlogs/vis'

config.input_size = 768

# max polygon per image
config.max_annotation = 200 #200

# max point per polygon
config.max_points = 20 #20

# use hard examples (annotated as '#')
config.use_hard = True

# demo tr threshold
config.tr_thresh = 0.5

# demo tcl threshold
config.tcl_thresh = 0.4

# expand ratio in post processing
config.post_process_expand = 0.3

# merge joined text instance when predicting
config.post_process_merge = False

# ── Embedding head / Discriminative Loss config ──
config.embedding_dim     = 8       # dimension of embedding vectors
config.delta_v           = 0.5     # pull margin (intra-instance)
config.delta_d           = 1.5     # push margin (inter-instance)
config.lambda_embed      = 0.1     # embedding loss weight — tune this first
config.embed_bandwidth   = 1.5     # MeanShift bandwidth at inference (= delta_d)
config.sigma_thresh      = 0.45    # above = split component via MeanShift
config.merge_thresh      = 0.5     # mean embedding distance below which to merge
config.use_embedding     = True    # False = fall back to contour-only (for A/B test)

def update_config(config, extra_config):
    for k, v in vars(extra_config).items():
        config[k] = v
    config.device = torch.device('cuda') if config.cuda else torch.device('cpu')

def print_config(config):
    print('==========Options============')
    for k, v in config.items():
        print('{}: {}'.format(k, v))
    print('=============End=============')
