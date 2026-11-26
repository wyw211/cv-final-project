from .config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.META_ARCHITECTURE = "Baseline"

_C.MODEL.FREEZE_LAYERS = []

# MoCo memory size
_C.MODEL.QUEUE_SIZE = 8192

# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
_C.MODEL.BACKBONE = CN()

_C.MODEL.BACKBONE.NAME = "build_resnet_backbone"
_C.MODEL.BACKBONE.DEPTH = "50x"
_C.MODEL.BACKBONE.LAST_STRIDE = 1
# Backbone feature dimension
_C.MODEL.BACKBONE.FEAT_DIM = 2048
# Normalization method for the convolution layers.
_C.MODEL.BACKBONE.NORM = "BN"
# If use IBN block in backbone
_C.MODEL.BACKBONE.WITH_IBN = False
# If use SE block in backbone
_C.MODEL.BACKBONE.WITH_SE = False
# If use Non-local block in backbone
_C.MODEL.BACKBONE.WITH_NL = False
# Vision Transformer options
_C.MODEL.BACKBONE.SIE_COE = 3.0
_C.MODEL.BACKBONE.STRIDE_SIZE = (16, 16)
_C.MODEL.BACKBONE.DROP_PATH_RATIO = 0.1
_C.MODEL.BACKBONE.DROP_RATIO = 0.0
_C.MODEL.BACKBONE.ATT_DROP_RATE = 0.0
# Recompute ViSA transformer blocks during backward; preserve upstream default.
_C.MODEL.BACKBONE.ACTIVATION_CHECKPOINTING = False
# If use ImageNet pretrain model
_C.MODEL.BACKBONE.PRETRAIN = False
# Pretrain model path
_C.MODEL.BACKBONE.PRETRAIN_PATH = ''

# num query in decoder
_C.MODEL.BACKBONE.INNER_SUB = False


# prompt
_C.MODEL.BACKBONE.USE_PROMPT = False
_C.MODEL.BACKBONE.USE_PRM = False
_C.MODEL.BACKBONE.PROMPT_LEN = 16
_C.MODEL.BACKBONE.PROMPT_DEPTH = 2
_C.MODEL.BACKBONE.PROMPT_SHIFT = 5.0
_C.MODEL.BACKBONE.PROMPT_SCALE = 20.0
_C.MODEL.BACKBONE.PROMPT_NAIVE_PROMPT = False
_C.MODEL.BACKBONE.EXPERTS = 8
_C.MODEL.BACKBONE.TOP_K = 2
_C.MODEL.BACKBONE.GRAPH_TOP_K = 64
# Expert-input fix (improvement). Defaults reproduce the official code exactly.
# "global": experts cross-attend to a single routing token (official; the
#           cross-attention is degenerate because softmax over one key is 1).
# "patch":  experts cross-attend to [routing token, patch tokens of b1].
_C.MODEL.BACKBONE.EXPERT_KV = "global"
# False: one top-k patch set chosen by the global token, shared by all queries
#        (official). True: each expert query selects its own top-k (paper Eq. 8).
_C.MODEL.BACKBONE.GRAPH_PER_QUERY = False
# False: both DLFM branches call share_gcn (official). True: the view-specific
#        branch uses its own view_gcn, as described in the paper.
_C.MODEL.BACKBONE.SEPARATE_VIEW_GCN = False
# v2 (expert attention). Std of the expert prompt-token init (official 0.02,
# i.e. |prompt| ~0.55 while LayerNorm'd keys are ~27, which makes the attention
# logits ~0 and the attention uniform).
_C.MODEL.BACKBONE.EXPERT_PROMPT_STD = 0.02
# Cosine (QK-normalised) expert cross-attention with a learnable temperature.
_C.MODEL.BACKBONE.EXPERT_QK_NORM = False
_C.MODEL.BACKBONE.EXPERT_QK_SCALE_INIT = 10.0
# v3 (router). Input of the aerial/ground routers: "view" = view token
# (official), "content" = mean of the b1 patch tokens of the image.
_C.MODEL.BACKBONE.ROUTER_INPUT = "view"
# Load-balancing loss: "official" = E * sum(mean_prob^2); "switch" = Switch
# Transformer E * sum(selected_fraction * mean_prob).
_C.MODEL.BACKBONE.MOE_BALANCE_LOSS = "official"
# Std of Gaussian noise added to router logits for top-k selection, training only.
_C.MODEL.BACKBONE.ROUTER_NOISE_STD = 0.0
# Weight of the load-balancing loss (official code hard-codes 0.0001; paper's
# best value is 0.001).
_C.MODEL.BACKBONE.MOE_BALANCE_LAMBDA = 0.0001


# ---------------------------------------------------------------------------- #
# REID HEADS options
# ---------------------------------------------------------------------------- #
_C.MODEL.HEADS = CN()
_C.MODEL.HEADS.NAME = "EmbeddingHead"
# Normalization method for the convolution layers.
_C.MODEL.HEADS.NORM = "BN"
# Number of identity
_C.MODEL.HEADS.NUM_CLASSES = 0
# Embedding dimension in head
_C.MODEL.HEADS.EMBEDDING_DIM = 0
# If use BNneck in embedding
_C.MODEL.HEADS.WITH_BNNECK = False
# Triplet feature using feature before(after) bnneck
_C.MODEL.HEADS.NECK_FEAT = "before"  # options: before, after
# Pooling layer type
_C.MODEL.HEADS.POOL_LAYER = "GlobalAvgPool"

# Classification layer type
_C.MODEL.HEADS.CLS_LAYER = "Linear"  # ArcSoftmax" or "CircleSoftmax"

# Margin and Scale for margin-based classification layer
_C.MODEL.HEADS.MARGIN = 0.
_C.MODEL.HEADS.SCALE = 1

# ---------------------------------------------------------------------------- #
# REID LOSSES options
# ---------------------------------------------------------------------------- #
_C.MODEL.LOSSES = CN()
_C.MODEL.LOSSES.NAME = ("CrossEntropyLoss",)

# Cross Entropy Loss options
_C.MODEL.LOSSES.CE = CN()
# if epsilon == 0, it means no label smooth regularization,
# if epsilon == -1, it means adaptive label smooth regularization
_C.MODEL.LOSSES.CE.EPSILON = 0.0
_C.MODEL.LOSSES.CE.ALPHA = 0.2
_C.MODEL.LOSSES.CE.SCALE = 1.0
# View-related Learning
_C.MODEL.LOSSES.CE.VIEW_ID = False
_C.MODEL.LOSSES.CE.VIEW_OREG = False
_C.MODEL.LOSSES.CE.VIEW_LAMBDA = 1.0


# Focal Loss options
_C.MODEL.LOSSES.FL = CN()
_C.MODEL.LOSSES.FL.ALPHA = 0.25
_C.MODEL.LOSSES.FL.GAMMA = 2
_C.MODEL.LOSSES.FL.SCALE = 1.0

# Triplet Loss options
_C.MODEL.LOSSES.TRI = CN()
_C.MODEL.LOSSES.TRI.MARGIN = 0.3
_C.MODEL.LOSSES.TRI.NORM_FEAT = False
_C.MODEL.LOSSES.TRI.HARD_MINING = False
_C.MODEL.LOSSES.TRI.SCALE = 1.0

# Circle Loss options
# View-aware prototype contrastive loss (P1, after ProAlign's PCL). Off by default.
# Per identity, momentum-updated aerial and ground prototypes; each feature is pulled
# to its own identity's prototype of the other view (cross) and of its own view (same).
# L3: illumination-consistency loss. NUM images of each batch are re-lit on the GPU
# (darkened if bright, brightened if dark) and their invariant feature is pulled to the
# original's (detached) by cosine distance. Training only.
_C.MODEL.LOSSES.ILLUM = CN()
_C.MODEL.LOSSES.ILLUM.ENABLED = False
_C.MODEL.LOSSES.ILLUM.WEIGHT = 1.0
_C.MODEL.LOSSES.ILLUM.NUM = 64

_C.MODEL.LOSSES.PCL = CN()
_C.MODEL.LOSSES.PCL.ENABLED = False
_C.MODEL.LOSSES.PCL.WEIGHT = 0.5        # per branch (invariant global + local)
_C.MODEL.LOSSES.PCL.TEMP = 0.05
_C.MODEL.LOSSES.PCL.MOMENTUM = 0.2      # weight of the old prototype
_C.MODEL.LOSSES.PCL.CROSS_WEIGHT = 0.5  # cross-view term weight; same-view gets 1 - this

_C.MODEL.LOSSES.CIRCLE = CN()
_C.MODEL.LOSSES.CIRCLE.MARGIN = 0.25
_C.MODEL.LOSSES.CIRCLE.GAMMA = 128
_C.MODEL.LOSSES.CIRCLE.SCALE = 1.0

# Cosface Loss options
_C.MODEL.LOSSES.COSFACE = CN()
_C.MODEL.LOSSES.COSFACE.MARGIN = 0.25
_C.MODEL.LOSSES.COSFACE.GAMMA = 128
_C.MODEL.LOSSES.COSFACE.SCALE = 1.0

# Path to a checkpoint file to be loaded to the model. You can find available models in the model zoo.
_C.MODEL.WEIGHTS = ""

# Values to be used for image normalization
_C.MODEL.PIXEL_MEAN = [0.485*255, 0.456*255, 0.406*255]
# Values to be used for image normalization
_C.MODEL.PIXEL_STD = [0.229*255, 0.224*255, 0.225*255]

_C.MODEL.USE_VISA = False
# -----------------------------------------------------------------------------
# KNOWLEDGE DISTILLATION
# -----------------------------------------------------------------------------

_C.KD = CN()
_C.KD.MODEL_CONFIG = []
_C.KD.MODEL_WEIGHTS = []
_C.KD.EMA = CN({"ENABLED": False})
_C.KD.EMA.MOMENTUM = 0.999

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.CAMERA = 0
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [256, 128]
# Size of the image during test
_C.INPUT.SIZE_TEST = [256, 128]

# `True` if cropping is used for data augmentation during training
_C.INPUT.CROP = CN({"ENABLED": False})
# Size of the image cropped
_C.INPUT.CROP.SIZE = [224, 224]
# Size of the origin size cropped
_C.INPUT.CROP.SCALE = [0.16, 1]
# Aspect ratio of the origin aspect ratio cropped
_C.INPUT.CROP.RATIO = [3./4., 4./3.]

# Random probability for image horizontal flip
_C.INPUT.FLIP = CN({"ENABLED": False})
_C.INPUT.FLIP.PROB = 0.5

# Value of padding size
_C.INPUT.PADDING = CN({"ENABLED": False})
_C.INPUT.PADDING.MODE = 'constant'
_C.INPUT.PADDING.SIZE = 10

# Random Rotation
# L1: low-light augmentation (training only), see transforms.RandomIllumination
_C.INPUT.LOWLIGHT = CN({"ENABLED": False})
_C.INPUT.LOWLIGHT.PROB = 0.5
_C.INPUT.LOWLIGHT.DARK_GAMMA = [2.0, 4.0]
_C.INPUT.LOWLIGHT.BRIGHT_GAMMA = [0.4, 0.75]
_C.INPUT.LOWLIGHT.CONTRAST = [0.3, 0.7]
_C.INPUT.LOWLIGHT.SATURATION = [0.3, 0.8]
_C.INPUT.LOWLIGHT.NOISE = 0.02
_C.INPUT.LOWLIGHT.DARK_THRESH = 0.30
_C.INPUT.LOWLIGHT.BRIGHT_THRESH = 0.38
# L2: deterministic brightness normalisation (train and test), see transforms.AutoGamma
_C.INPUT.AUTOGAMMA = CN({"ENABLED": False})
_C.INPUT.AUTOGAMMA.TARGET = 0.45
_C.INPUT.AUTOGAMMA.MIN_GAMMA = 0.4

_C.INPUT.ROTATION = CN({"ENABLED": False})
_C.INPUT.ROTATION.DEGREES = 30

# Random color jitter
_C.INPUT.CJ = CN({"ENABLED": False})
_C.INPUT.CJ.PROB = 0.5
_C.INPUT.CJ.BRIGHTNESS = 0.15
_C.INPUT.CJ.CONTRAST = 0.15
_C.INPUT.CJ.SATURATION = 0.1
_C.INPUT.CJ.HUE = 0.1

# Random Affine
_C.INPUT.AFFINE = CN({"ENABLED": False})

# Auto augmentation
_C.INPUT.AUTOAUG = CN({"ENABLED": False})
_C.INPUT.AUTOAUG.PROB = 0.0

# Augmix augmentation
_C.INPUT.AUGMIX = CN({"ENABLED": False})
_C.INPUT.AUGMIX.PROB = 0.0

# Random Erasing
_C.INPUT.REA = CN({"ENABLED": False})
_C.INPUT.REA.PROB = 0.5
_C.INPUT.REA.VALUE = [0.485*255, 0.456*255, 0.406*255]
# Random Patch
_C.INPUT.RPT = CN({"ENABLED": False})
_C.INPUT.RPT.PROB = 0.5

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training
_C.DATASETS.NAMES = ("Market1501",)
# List of the dataset names for testing
_C.DATASETS.TESTS = ("Market1501",)
# Combine trainset and testset joint training
_C.DATASETS.COMBINEALL = False

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Options: TrainingSampler, NaiveIdentitySampler, BalancedIdentitySampler
_C.DATALOADER.SAMPLER_TRAIN = "TrainingSampler"
# Number of instance for each person
_C.DATALOADER.NUM_INSTANCE = 4
_C.DATALOADER.NUM_WORKERS = 8

# For set re-weight
_C.DATALOADER.SET_WEIGHT = []

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()

# AUTOMATIC MIXED PRECISION
_C.SOLVER.AMP = CN({"ENABLED": False})

# Optimizer
_C.SOLVER.OPT = "Adam"

_C.SOLVER.MAX_EPOCH = 120

_C.SOLVER.BASE_LR = 3e-4

# This LR is applied to the last classification layer if
# you want to 10x higher than BASE_LR.
_C.SOLVER.HEADS_LR_FACTOR = 1.
# LR multiplier for parameters under these module-name prefixes (pretrained ViT and
# its copied last block). 1.0 = official. Used by the PersonViT (P2) pilots.
_C.SOLVER.PRETRAINED_LR_FACTOR = 1.0
_C.SOLVER.PRETRAINED_PREFIXES = ["backbone.base.", "backbone.b1."]

_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.NESTEROV = False

_C.SOLVER.WEIGHT_DECAY = 0.0005
# The weight decay that's applied to parameters of normalization layers
# (typically the affine transformation)
_C.SOLVER.WEIGHT_DECAY_NORM = 0.0005

# The previous detection code used a 2x higher LR and 0 WD for bias.
# This is not useful (at least for recent models). You should avoid
# changing these and they exists only to reproduce previous model
# training if desired.
_C.SOLVER.BIAS_LR_FACTOR = 1.0
_C.SOLVER.WEIGHT_DECAY_BIAS = _C.SOLVER.WEIGHT_DECAY

# Multi-step learning rate options
_C.SOLVER.SCHED = "MultiStepLR"

_C.SOLVER.DELAY_EPOCHS = 0

_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = [30, 55]

# Cosine annealing learning rate options
_C.SOLVER.ETA_MIN_LR = 1e-7

# Warmup options
_C.SOLVER.WARMUP_FACTOR = 0.1
_C.SOLVER.WARMUP_ITERS = 1000
_C.SOLVER.WARMUP_METHOD = "linear"

# Backbone freeze iters
_C.SOLVER.FREEZE_ITERS = 0

_C.SOLVER.CHECKPOINT_PERIOD = 20

# Number of images per batch across all machines.
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 256, each GPU will
# see 32 images per batch
_C.SOLVER.IMS_PER_BATCH = 64

# Gradient clipping
_C.SOLVER.CLIP_GRADIENTS = CN({"ENABLED": False})
# Type of gradient clipping, currently 2 values are supported:
# - "value": the absolute values of elements of each gradients are clipped
# - "norm": the norm of the gradient for each parameter is clipped thus
#   affecting all elements in the parameter
_C.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"
# Maximum absolute value used for clipping gradients
_C.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 5.0
# Floating point number p for L-p norm to be used with the "norm"
# gradient clipping type; for L-inf, please specify .inf
_C.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0

_C.TEST = CN()

_C.TEST.EVAL_PERIOD = 20

# Number of images per batch across all machines.
_C.TEST.IMS_PER_BATCH = 64
# DataLoader workers for evaluation (upstream hard-coded 4). On Windows every
# worker commits ~6 GB of virtual memory; 0 loads in the main process and
# avoids exhausting the commit limit while the 8 training workers are alive.
# Affects only evaluation speed, not results.
_C.TEST.NUM_WORKERS = 4
_C.TEST.METRIC = "cosine"
_C.TEST.ROC = CN({"ENABLED": False})
_C.TEST.FLIP = CN({"ENABLED": False})

# Average query expansion
_C.TEST.AQE = CN({"ENABLED": False})
_C.TEST.AQE.ALPHA = 3.0
_C.TEST.AQE.QE_TIME = 1
_C.TEST.AQE.QE_K = 5

# Re-rank
_C.TEST.RERANK = CN({"ENABLED": False})
_C.TEST.RERANK.K1 = 20
_C.TEST.RERANK.K2 = 6
_C.TEST.RERANK.LAMBDA = 0.3

# Precise batchnorm
_C.TEST.PRECISE_BN = CN({"ENABLED": False})
_C.TEST.PRECISE_BN.DATASET = 'Market1501'
_C.TEST.PRECISE_BN.NUM_ITER = 300

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "logs/"

# Global RNG seed (upstream hard-codes 0 in default_setup). Different values
# change model init, sampler order and augmentation; used for seed-variance runs.
_C.SEED = 0

# Benchmark different cudnn algorithms.
# If input images have very different sizes, this option will have large overhead
# for about 10k iterations. It usually hurts total time, but can benefit for certain models.
# If input images have the same or similar sizes, benchmark is often helpful.
_C.CUDNN_BENCHMARK = False
