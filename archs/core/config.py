from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import yaml
from easydict import EasyDict as edict

config = edict()

config.WORKERS = 8
config.LOG_DIR = ''
config.MODEL_DIR = ''
config.RESULT_DIR = ''
config.DATA_DIR = ''
config.VERBOSE = False

# CUDNN related params
config.CUDNN = edict()
config.CUDNN.BENCHMARK = True
config.CUDNN.DETERMINISTIC = False
config.CUDNN.ENABLED = True

# DTG_SPL related params
config.DTG_SPL = edict()
config.DTG_SPL.VIDEO_EMBEDDER = edict()
config.DTG_SPL.VIDEO_EMBEDDER.NAME = ''
config.DTG_SPL.VIDEO_EMBEDDER.PARAMS = None
config.DTG_SPL.QUERY_EMBEDDER = edict()
config.DTG_SPL.QUERY_EMBEDDER.NAME = ''
config.DTG_SPL.QUERY_EMBEDDER.PARAMS = None
config.DTG_SPL.REGRESSION = edict()
config.DTG_SPL.REGRESSION.NAME = ''
config.DTG_SPL.REGRESSION.PARAMS = None
config.DTG_SPL.TRANSFORMER = edict()
config.DTG_SPL.TRANSFORMER.NAME = ''
config.DTG_SPL.TRANSFORMER.PARAMS = None
config.DTG_SPL.SEMANTIC = edict()
config.DTG_SPL.SEMANTIC.NAME = ''
config.DTG_SPL.SEMANTIC.PARAMS = None
config.DTG_SPL.MATCHING = edict()
config.DTG_SPL.MATCHING.NAME = ''
config.DTG_SPL.MATCHING.PARAMS = None
# common params for NETWORK
config.MODEL = edict()
config.MODEL.NAME = ''
config.MODEL.CHECKPOINT = '' # The checkpoint for the best performance

# DATASET related params
config.DATASET = edict()
config.DATASET.ROOT = ''
config.DATASET.NAME = ''
config.DATASET.VIS_INPUT_TYPE = ''
config.DATASET.NO_VAL = False
config.DATASET.NUM_SAMPLE_CLIPS = 128
config.DATASET.SPLIT = ''
config.DATASET.NORMALIZE = False

# train
config.TRAIN = edict()
config.TRAIN.LR = 0.001
config.TRAIN.WEIGHT_DECAY = 0
config.TRAIN.FACTOR = 0.8
config.TRAIN.PATIENCE = 20
config.TRAIN.MAX_EPOCH = 80
config.TRAIN.BATCH_SIZE = 4
config.TRAIN.SHUFFLE = True
config.TRAIN.CONTINUE = False
config.TRAIN.STEPS = [15]
config.TRAIN.INTRA_CORRELATION = False

config.LOSS = edict()
config.LOSS.LAMBDA = 0.02
config.LOSS.GAMMA1 = 1
config.LOSS.GAMMA2 = 1
config.LOSS.EPR = 5.0
# test
config.TEST = edict()
config.TEST.RECALL = []
config.TEST.TIOU = []
config.TEST.NMS_THRESH = 1
config.TEST.INTERVAL = 1
config.TEST.EVAL_TRAIN = False
config.TEST.BATCH_SIZE = 1
config.TEST.EVAL_SIZE = 1
config.TEST.TOP_K = 10
config.TEST.EVAL_BETA = 0.5
config.TEST.EVAL_SENT = True
config.TEST.NMS_TYPE = ''
config.TEST.T_THRESH = 0.5

def _update_dict(cfg, value):
    for k, v in value.items():
        if k in cfg:
            if k == 'PARAMS':
                cfg[k] = v
            elif isinstance(v, dict):
                _update_dict(cfg[k],v)
            else:
                cfg[k] = v
        else:
            raise ValueError("{} not exist in config.py".format(k))

def update_config(config_file):
    with open(config_file) as f:
        exp_config = edict(yaml.load(f, Loader=yaml.FullLoader))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    _update_dict(config[k], v)
                else:
                    config[k] = v
            else:
                raise ValueError("{} not exist in config.py".format(k))
