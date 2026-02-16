import torch


cfg_blurs = {1: 0, 2: 0, 3: 0, 4: 4, 5: 8, 6: 16, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0,
             17: 4, 18: 8, 19: 16}

configurations = {
    # 1 - original.
    # 2 - debug (1 gpu)
    # 3 - Blur 0
    # 4 - Blur 4
    # 5 - Blur 8
    # 6 - Blur 16
    # 7 - Blur 0 (same as 3, with suffix '_rep').
    # 8 - Blur 0 (same as 3, with suffix '_printLR2').
    # 9 - Blur 0 (same as 3, with suffix '_use_clamp'). Ran this job after changing code in
    #       head.metrics.ArcFace.forward: added clamp, to avoid cosine values > 1 (try to solve nan loss issue 4/2/26).
    # 10 - Blur 0 (copied config 4 which yeilded good training results, but cfg_blurs[10] = 10 so 'IR_SE_50_ArcFace_Blur0'
    # 11 - Blur 0 (copied config 4 but with suffix try_not_open_out). Accidantly didn't change BLUR, so blur=4.
    # 12 - Blur 0, 2 workers (copied config 11, suffix: 2workers, NUM_WORKERS = 2).
    # 13 - Blur 0, copied config 3, but new mean & std rgb values (according to get_dataset_stats). suffix: "_norm_rgb".
    # 14 - Blur 0, norm rgb + 2 workers (copied config 13, suffix: norm_rgb_2workers, NUM_WORKERS = 2).
    # 15 - Blur 0, 4 workers (copied config 12, suffix: "_norm_rgb", NUM_WORKERS = 4).
    # 16 - Blur 0, 2 workers, 1 gpu (copied config 12, suffix: "_2wrkrs_1gpu", NUM_WORKERS = 2, GPU_ID = [0]).
    #       need to ask for 70GB to get A100.
    # 17 - Blur 4 (copied config 12, but cfg_blurs[17]=4, so suffix "blur4_2workers").
    # 18 - Blur 8 (copied config 12, but cfg_blurs[18]=8, so suffix "blur8_2workers").
    # 19 - Blur 16 (copied config 12, but cfg_blurs[19]=16, so suffix "blur16_2workers").

    1: dict(
        SEED = 1337, # random seed for reproduce results

        DATA_ROOT = '/home/peter/Project/face.evoLVe.PyTorch/data', # the parent root where your train/val/test data are stored
        MODEL_ROOT = '/home/peter/Project/face.evoLVe.PyTorch/model', # the root to buffer your checkpoints
        LOG_ROOT = '/home/peter/Project/face.evoLVe.PyTorch/log', # the root to log your train/val status
        BACKBONE_RESUME_ROOT = './', # the root to resume training from a saved checkpoint
        HEAD_RESUME_ROOT = './', # the root to resume training from a saved checkpoint

        BACKBONE_NAME = 'IR_SE_50', # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
        HEAD_NAME = 'ArcFace', # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
        LOSS_NAME = 'Focal', # support: ['Focal', 'Softmax']

        INPUT_SIZE = [112, 112], # support: [112, 112] and [224, 224]
        RGB_MEAN = [0.5, 0.5, 0.5], # for normalize inputs to [-1, 1]
        RGB_STD = [0.5, 0.5, 0.5],
        EMBEDDING_SIZE = 512, # feature dimension
        BATCH_SIZE = 512,
        DROP_LAST = True, # whether drop the last batch to ensure consistent batch_norm statistics
        LR = 0.1, # initial LR
        NUM_EPOCH = 125, # total epoch number (use the firt 1/25 epochs to warm up)
        WEIGHT_DECAY = 5e-4, # do not apply to batch_norm parameters
        MOMENTUM = 0.9,
        STAGES = [35, 65, 95], # epoch stages to decay learning rate

        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        MULTI_GPU = True, # flag to use multiple GPUs; if you choose to train with single GPU, you should first run "export CUDA_VISILE_DEVICES=device_id" to specify the GPU card you want to use
        GPU_ID = [0, 1, 2, 3], # specify your GPU ids
        PIN_MEMORY = True,
        NUM_WORKERS = 0,
    ),

    2: dict(
        SEED = 1337, # random seed for reproduce results

        DATA_ROOT = '/home/projects/bagon/ilanaveh/data', # the parent root where your train/val/test data are stored
        MODEL_ROOT = '/home/projects/bagon/ilanaveh/code/face.evoLVe/model', # the root to buffer your checkpoints
        LOG_ROOT = f'/home/projects/bagon/ilanaveh/code/face.evoLVe/log/IR_SE_50_ArcFace_Blur{cfg_blurs[2]}_db', # the root to log your train/val status
        BACKBONE_RESUME_ROOT = f'/home/projects/bagon/ilanaveh/code/face.evoLVe/model/IR_SE_50_ArcFace_Blur{cfg_blurs[2]}_db', # the root to resume training from a saved checkpoint
        HEAD_RESUME_ROOT = f'/home/projects/bagon/ilanaveh/code/face.evoLVe/model/IR_SE_50_ArcFace_Blur{cfg_blurs[2]}_db', # the root to resume training from a saved checkpoint

        BACKBONE_NAME = 'IR_SE_50', # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
        HEAD_NAME = 'ArcFace', # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
        LOSS_NAME = 'Focal', # support: ['Focal', 'Softmax']

        INPUT_SIZE = [112, 112], # support: [112, 112] and [224, 224]
        RGB_MEAN = [0.5, 0.5, 0.5], # for normalize inputs to [-1, 1]
        RGB_STD = [0.5, 0.5, 0.5],
        EMBEDDING_SIZE = 512, # feature dimension
        BATCH_SIZE = 512,
        DROP_LAST = True, # whether drop the last batch to ensure consistent batch_norm statistics
        LR = 0.1, # initial LR
        NUM_EPOCH = 120, # total epoch number (use the firt 1/25 epochs to warm up)  *-- change from 125
        WEIGHT_DECAY = 5e-4, # do not apply to batch_norm parameters
        MOMENTUM = 0.9,
        STAGES = [30, 60, 90], # epoch stages to decay learning rate  *-- Change from [35, 65, 95]

        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        MULTI_GPU = True, # flag to use multiple GPUs; if you choose to train with single GPU, you should first run "export CUDA_VISILE_DEVICES=device_id" to specify the GPU card you want to use
        GPU_ID = [0], # specify your GPU ids  *-- Original: [0, 1, 2, 3]; for debugging: [0]
        PIN_MEMORY = True,
        NUM_WORKERS = 0,
        BLUR = cfg_blurs[2],
    ),

    3: dict(
        SEED = 1337, # random seed for reproduce results

        DATA_ROOT = '/home/projects/bagon/ilanaveh/data', # the parent root where your train/val/test data are stored
        MODEL_ROOT = '/home/projects/bagon/ilanaveh/code/face.evoLVe/model', # the root to buffer your checkpoints
        LOG_ROOT = f'/home/projects/bagon/ilanaveh/code/face.evoLVe/log/IR_SE_50_ArcFace_Blur{cfg_blurs[3]}', # the root to log your train/val status
        BACKBONE_RESUME_ROOT = f'/home/projects/bagon/ilanaveh/code/face.evoLVe/model/IR_SE_50_ArcFace_Blur{cfg_blurs[3]}', # the root to resume training from a saved checkpoint
        HEAD_RESUME_ROOT = f'/home/projects/bagon/ilanaveh/code/face.evoLVe/model/IR_SE_50_ArcFace_Blur{cfg_blurs[3]}', # the root to resume training from a saved checkpoint

        BACKBONE_NAME = 'IR_SE_50', # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
        HEAD_NAME = 'ArcFace', # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
        LOSS_NAME = 'Focal', # support: ['Focal', 'Softmax']

        INPUT_SIZE = [112, 112], # support: [112, 112] and [224, 224]
        RGB_MEAN = [0.5, 0.5, 0.5], # for normalize inputs to [-1, 1]
        RGB_STD = [0.5, 0.5, 0.5],
        EMBEDDING_SIZE = 512, # feature dimension
        BATCH_SIZE = 512,
        DROP_LAST = True, # whether drop the last batch to ensure consistent batch_norm statistics
        LR = 0.1, # initial LR
        NUM_EPOCH = 120, # total epoch number (use the firt 1/25 epochs to warm up)  *-- change from 125
        WEIGHT_DECAY = 5e-4, # do not apply to batch_norm parameters
        MOMENTUM = 0.9,
        STAGES = [30, 60, 90], # epoch stages to decay learning rate  *-- Change from [35, 65, 95]

        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        MULTI_GPU = True, # flag to use multiple GPUs; if you choose to train with single GPU, you should first run "export CUDA_VISILE_DEVICES=device_id" to specify the GPU card you want to use
        GPU_ID = [0, 1, 2, 3], # specify your GPU ids  *-- Original: [0, 1, 2, 3]; for debugging - change to [0]
        PIN_MEMORY = True,
        NUM_WORKERS = 0,
        BLUR = cfg_blurs[3],
    ),

    4: dict(
        SEED = 1337, # random seed for reproduce results

        DATA_ROOT = '/home/projects/bagon/ilanaveh/data', # the parent root where your train/val/test data are stored
        MODEL_ROOT = '/home/projects/bagon/ilanaveh/code/face.evoLVe/model', # the root to buffer your checkpoints
        LOG_ROOT = f'/home/projects/bagon/ilanaveh/code/face.evoLVe/log/IR_SE_50_ArcFace_Blur{cfg_blurs[4]}', # the root to log your train/val status
        BACKBONE_RESUME_ROOT = f'/home/projects/bagon/ilanaveh/code/face.evoLVe/model/IR_SE_50_ArcFace_Blur{cfg_blurs[4]}', # the root to resume training from a saved checkpoint
        HEAD_RESUME_ROOT = f'/home/projects/bagon/ilanaveh/code/face.evoLVe/model/IR_SE_50_ArcFace_Blur{cfg_blurs[4]}', # the root to resume training from a saved checkpoint

        BACKBONE_NAME = 'IR_SE_50', # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
        HEAD_NAME = 'ArcFace', # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
        LOSS_NAME = 'Focal', # support: ['Focal', 'Softmax']

        INPUT_SIZE = [112, 112], # support: [112, 112] and [224, 224]
        RGB_MEAN = [0.5, 0.5, 0.5], # for normalize inputs to [-1, 1]
        RGB_STD = [0.5, 0.5, 0.5],
        EMBEDDING_SIZE = 512, # feature dimension
        BATCH_SIZE = 512,
        DROP_LAST = True, # whether drop the last batch to ensure consistent batch_norm statistics
        LR = 0.1, # initial LR
        NUM_EPOCH = 120, # total epoch number (use the firt 1/25 epochs to warm up)  *-- change from 125
        WEIGHT_DECAY = 5e-4, # do not apply to batch_norm parameters
        MOMENTUM = 0.9,
        STAGES = [30, 60, 90], # epoch stages to decay learning rate  *-- Change from [35, 65, 95]

        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        MULTI_GPU = True, # flag to use multiple GPUs; if you choose to train with single GPU, you should first run "export CUDA_VISILE_DEVICES=device_id" to specify the GPU card you want to use
        GPU_ID = [0, 1, 2, 3], # specify your GPU ids  *-- Original: [0, 1, 2, 3]; for debugging - change to [0]
        PIN_MEMORY = True,
        NUM_WORKERS = 0,
        BLUR = cfg_blurs[4],
    ),

    5: dict(
        SEED = 1337, # random seed for reproduce results

        DATA_ROOT = '/home/projects/bagon/ilanaveh/data', # the parent root where your train/val/test data are stored
        MODEL_ROOT = '/home/projects/bagon/ilanaveh/code/face.evoLVe/model', # the root to buffer your checkpoints
        LOG_ROOT = f'/home/projects/bagon/ilanaveh/code/face.evoLVe/log/IR_SE_50_ArcFace_Blur{cfg_blurs[5]}', # the root to log your train/val status
        BACKBONE_RESUME_ROOT = f'/home/projects/bagon/ilanaveh/code/face.evoLVe/model/IR_SE_50_ArcFace_Blur{cfg_blurs[5]}', # the root to resume training from a saved checkpoint
        HEAD_RESUME_ROOT = f'/home/projects/bagon/ilanaveh/code/face.evoLVe/model/IR_SE_50_ArcFace_Blur{cfg_blurs[5]}', # the root to resume training from a saved checkpoint

        BACKBONE_NAME = 'IR_SE_50', # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
        HEAD_NAME = 'ArcFace', # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
        LOSS_NAME = 'Focal', # support: ['Focal', 'Softmax']

        INPUT_SIZE = [112, 112], # support: [112, 112] and [224, 224]
        RGB_MEAN = [0.5, 0.5, 0.5], # for normalize inputs to [-1, 1]
        RGB_STD = [0.5, 0.5, 0.5],
        EMBEDDING_SIZE = 512, # feature dimension
        BATCH_SIZE = 512,
        DROP_LAST = True, # whether drop the last batch to ensure consistent batch_norm statistics
        LR = 0.1, # initial LR
        NUM_EPOCH = 120, # total epoch number (use the firt 1/25 epochs to warm up)  *-- change from 125
        WEIGHT_DECAY = 5e-4, # do not apply to batch_norm parameters
        MOMENTUM = 0.9,
        STAGES = [30, 60, 90], # epoch stages to decay learning rate  *-- Change from [35, 65, 95]

        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        MULTI_GPU = True, # flag to use multiple GPUs; if you choose to train with single GPU, you should first run "export CUDA_VISILE_DEVICES=device_id" to specify the GPU card you want to use
        GPU_ID = [0, 1, 2, 3], # specify your GPU ids  *-- Original: [0, 1, 2, 3]; for debugging - change to [0]
        PIN_MEMORY = True,
        NUM_WORKERS = 0,
        BLUR = cfg_blurs[5],
    ),

    6: dict(
        SEED = 1337, # random seed for reproduce results

        DATA_ROOT = '/home/projects/bagon/ilanaveh/data', # the parent root where your train/val/test data are stored
        MODEL_ROOT = '/home/projects/bagon/ilanaveh/code/face.evoLVe/model', # the root to buffer your checkpoints
        LOG_ROOT = f'/home/projects/bagon/ilanaveh/code/face.evoLVe/log/IR_SE_50_ArcFace_Blur{cfg_blurs[6]}', # the root to log your train/val status
        BACKBONE_RESUME_ROOT = f'/home/projects/bagon/ilanaveh/code/face.evoLVe/model/IR_SE_50_ArcFace_Blur{cfg_blurs[6]}', # the root to resume training from a saved checkpoint
        HEAD_RESUME_ROOT = f'/home/projects/bagon/ilanaveh/code/face.evoLVe/model/IR_SE_50_ArcFace_Blur{cfg_blurs[6]}', # the root to resume training from a saved checkpoint

        BACKBONE_NAME = 'IR_SE_50', # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
        HEAD_NAME = 'ArcFace', # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
        LOSS_NAME = 'Focal', # support: ['Focal', 'Softmax']

        INPUT_SIZE = [112, 112], # support: [112, 112] and [224, 224]
        RGB_MEAN = [0.5, 0.5, 0.5], # for normalize inputs to [-1, 1]
        RGB_STD = [0.5, 0.5, 0.5],
        EMBEDDING_SIZE = 512, # feature dimension
        BATCH_SIZE = 512,
        DROP_LAST = True, # whether drop the last batch to ensure consistent batch_norm statistics
        LR = 0.1, # initial LR
        NUM_EPOCH = 120, # total epoch number (use the firt 1/25 epochs to warm up)  *-- change from 125
        WEIGHT_DECAY = 5e-4, # do not apply to batch_norm parameters
        MOMENTUM = 0.9,
        STAGES = [30, 60, 90], # epoch stages to decay learning rate  *-- Change from [35, 65, 95]

        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        MULTI_GPU = True, # flag to use multiple GPUs; if you choose to train with single GPU, you should first run "export CUDA_VISILE_DEVICES=device_id" to specify the GPU card you want to use
        GPU_ID = [0, 1, 2, 3], # specify your GPU ids  *-- Original: [0, 1, 2, 3]; for debugging - change to [0]
        PIN_MEMORY = True,
        NUM_WORKERS = 0,
        BLUR = cfg_blurs[6],
    ),

    7: dict(
        SEED = 1337, # random seed for reproduce results

        DATA_ROOT = '/home/projects/bagon/ilanaveh/data', # the parent root where your train/val/test data are stored
        MODEL_ROOT = '/home/projects/bagon/ilanaveh/code/face.evoLVe/model', # the root to buffer your checkpoints
        LOG_ROOT = f'/home/projects/bagon/ilanaveh/code/face.evoLVe/log/IR_SE_50_ArcFace_Blur{cfg_blurs[7]}_rep', # the root to log your train/val status
        BACKBONE_RESUME_ROOT = f'/home/projects/bagon/ilanaveh/code/face.evoLVe/model/IR_SE_50_ArcFace_Blur{cfg_blurs[7]}_rep', # the root to resume training from a saved checkpoint
        HEAD_RESUME_ROOT = f'/home/projects/bagon/ilanaveh/code/face.evoLVe/model/IR_SE_50_ArcFace_Blur{cfg_blurs[7]}_rep', # the root to resume training from a saved checkpoint

        BACKBONE_NAME = 'IR_SE_50', # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
        HEAD_NAME = 'ArcFace', # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
        LOSS_NAME = 'Focal', # support: ['Focal', 'Softmax']

        INPUT_SIZE = [112, 112], # support: [112, 112] and [224, 224]
        RGB_MEAN = [0.5, 0.5, 0.5], # for normalize inputs to [-1, 1]
        RGB_STD = [0.5, 0.5, 0.5],
        EMBEDDING_SIZE = 512, # feature dimension
        BATCH_SIZE = 512,
        DROP_LAST = True, # whether drop the last batch to ensure consistent batch_norm statistics
        LR = 0.1, # initial LR
        NUM_EPOCH = 120, # total epoch number (use the firt 1/25 epochs to warm up)  *-- change from 125
        WEIGHT_DECAY = 5e-4, # do not apply to batch_norm parameters
        MOMENTUM = 0.9,
        STAGES = [30, 60, 90], # epoch stages to decay learning rate  *-- Change from [35, 65, 95]

        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        MULTI_GPU = True, # flag to use multiple GPUs; if you choose to train with single GPU, you should first run "export CUDA_VISILE_DEVICES=device_id" to specify the GPU card you want to use
        GPU_ID = [0, 1, 2, 3], # specify your GPU ids  *-- Original: [0, 1, 2, 3]; for debugging - change to [0]
        PIN_MEMORY = True,
        NUM_WORKERS = 0,
        BLUR = cfg_blurs[7],
    ),

    8: dict(
        SEED = 1337, # random seed for reproduce results

        DATA_ROOT = '/home/projects/bagon/ilanaveh/data', # the parent root where your train/val/test data are stored
        MODEL_ROOT = '/home/projects/bagon/ilanaveh/code/face.evoLVe/model', # the root to buffer your checkpoints
        LOG_ROOT = f'/home/projects/bagon/ilanaveh/code/face.evoLVe/log/IR_SE_50_ArcFace_Blur{cfg_blurs[8]}_printLR2', # the root to log your train/val status
        BACKBONE_RESUME_ROOT = f'/home/projects/bagon/ilanaveh/code/face.evoLVe/model/IR_SE_50_ArcFace_Blur{cfg_blurs[8]}_printLR2', # the root to resume training from a saved checkpoint
        HEAD_RESUME_ROOT = f'/home/projects/bagon/ilanaveh/code/face.evoLVe/model/IR_SE_50_ArcFace_Blur{cfg_blurs[8]}_printLR2', # the root to resume training from a saved checkpoint

        BACKBONE_NAME = 'IR_SE_50', # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
        HEAD_NAME = 'ArcFace', # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
        LOSS_NAME = 'Focal', # support: ['Focal', 'Softmax']

        INPUT_SIZE = [112, 112], # support: [112, 112] and [224, 224]
        RGB_MEAN = [0.5, 0.5, 0.5], # for normalize inputs to [-1, 1]
        RGB_STD = [0.5, 0.5, 0.5],
        EMBEDDING_SIZE = 512, # feature dimension
        BATCH_SIZE = 512,
        DROP_LAST = True, # whether drop the last batch to ensure consistent batch_norm statistics
        LR = 0.1, # initial LR
        NUM_EPOCH = 120, # total epoch number (use the firt 1/25 epochs to warm up)  *-- change from 125
        WEIGHT_DECAY = 5e-4, # do not apply to batch_norm parameters
        MOMENTUM = 0.9,
        STAGES = [30, 60, 90], # epoch stages to decay learning rate  *-- Change from [35, 65, 95]

        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        MULTI_GPU = True, # flag to use multiple GPUs; if you choose to train with single GPU, you should first run "export CUDA_VISILE_DEVICES=device_id" to specify the GPU card you want to use
        GPU_ID = [0, 1, 2, 3], # specify your GPU ids  *-- Original: [0, 1, 2, 3]; for debugging - change to [0]
        PIN_MEMORY = True,
        NUM_WORKERS = 0,
        BLUR = cfg_blurs[8],
    ),

    9: dict(
        SEED = 1337, # random seed for reproduce results

        DATA_ROOT = '/home/projects/bagon/ilanaveh/data', # the parent root where your train/val/test data are stored
        MODEL_ROOT = '/home/projects/bagon/ilanaveh/code/face.evoLVe/model', # the root to buffer your checkpoints
        LOG_ROOT = f'/home/projects/bagon/ilanaveh/code/face.evoLVe/log/IR_SE_50_ArcFace_Blur{cfg_blurs[9]}_use_clamp', # the root to log your train/val status
        BACKBONE_RESUME_ROOT = f'/home/projects/bagon/ilanaveh/code/face.evoLVe/model/IR_SE_50_ArcFace_Blur{cfg_blurs[9]}_use_clamp', # the root to resume training from a saved checkpoint
        HEAD_RESUME_ROOT = f'/home/projects/bagon/ilanaveh/code/face.evoLVe/model/IR_SE_50_ArcFace_Blur{cfg_blurs[9]}_use_clamp', # the root to resume training from a saved checkpoint

        BACKBONE_NAME = 'IR_SE_50', # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
        HEAD_NAME = 'ArcFace', # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
        LOSS_NAME = 'Focal', # support: ['Focal', 'Softmax']

        INPUT_SIZE = [112, 112], # support: [112, 112] and [224, 224]
        RGB_MEAN = [0.5, 0.5, 0.5], # for normalize inputs to [-1, 1]
        RGB_STD = [0.5, 0.5, 0.5],
        EMBEDDING_SIZE = 512, # feature dimension
        BATCH_SIZE = 512,
        DROP_LAST = True, # whether drop the last batch to ensure consistent batch_norm statistics
        LR = 0.1, # initial LR
        NUM_EPOCH = 120, # total epoch number (use the firt 1/25 epochs to warm up)  *-- change from 125
        WEIGHT_DECAY = 5e-4, # do not apply to batch_norm parameters
        MOMENTUM = 0.9,
        STAGES = [30, 60, 90], # epoch stages to decay learning rate  *-- Change from [35, 65, 95]

        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        MULTI_GPU = True, # flag to use multiple GPUs; if you choose to train with single GPU, you should first run "export CUDA_VISILE_DEVICES=device_id" to specify the GPU card you want to use
        GPU_ID = [0, 1, 2, 3], # specify your GPU ids  *-- Original: [0, 1, 2, 3]; for debugging - change to [0]
        PIN_MEMORY = True,
        NUM_WORKERS = 0,
        BLUR = cfg_blurs[9],
    ),

    10: dict(
        SEED = 1337, # random seed for reproduce results

        DATA_ROOT = '/home/projects/bagon/ilanaveh/data', # the parent root where your train/val/test data are stored
        MODEL_ROOT = '/home/projects/bagon/ilanaveh/code/face.evoLVe/model', # the root to buffer your checkpoints
        LOG_ROOT = f'/home/projects/bagon/ilanaveh/code/face.evoLVe/log/IR_SE_50_ArcFace_Blur{cfg_blurs[10]}', # the root to log your train/val status
        BACKBONE_RESUME_ROOT = f'/home/projects/bagon/ilanaveh/code/face.evoLVe/model/IR_SE_50_ArcFace_Blur{cfg_blurs[10]}', # the root to resume training from a saved checkpoint
        HEAD_RESUME_ROOT = f'/home/projects/bagon/ilanaveh/code/face.evoLVe/model/IR_SE_50_ArcFace_Blur{cfg_blurs[10]}', # the root to resume training from a saved checkpoint

        BACKBONE_NAME = 'IR_SE_50', # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
        HEAD_NAME = 'ArcFace', # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
        LOSS_NAME = 'Focal', # support: ['Focal', 'Softmax']

        INPUT_SIZE = [112, 112], # support: [112, 112] and [224, 224]
        RGB_MEAN = [0.5, 0.5, 0.5], # for normalize inputs to [-1, 1]
        RGB_STD = [0.5, 0.5, 0.5],
        EMBEDDING_SIZE = 512, # feature dimension
        BATCH_SIZE = 512,
        DROP_LAST = True, # whether drop the last batch to ensure consistent batch_norm statistics
        LR = 0.1, # initial LR
        NUM_EPOCH = 120, # total epoch number (use the firt 1/25 epochs to warm up)  *-- change from 125
        WEIGHT_DECAY = 5e-4, # do not apply to batch_norm parameters
        MOMENTUM = 0.9,
        STAGES = [30, 60, 90], # epoch stages to decay learning rate  *-- Change from [35, 65, 95]

        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        MULTI_GPU = True, # flag to use multiple GPUs; if you choose to train with single GPU, you should first run "export CUDA_VISILE_DEVICES=device_id" to specify the GPU card you want to use
        GPU_ID = [0, 1, 2, 3], # specify your GPU ids  *-- Original: [0, 1, 2, 3]; for debugging - change to [0]
        PIN_MEMORY = True,
        NUM_WORKERS = 0,
        BLUR = cfg_blurs[10],
    ),

    11: dict(
        SEED = 1337, # random seed for reproduce results

        DATA_ROOT = '/home/projects/bagon/ilanaveh/data', # the parent root where your train/val/test data are stored
        MODEL_ROOT = '/home/projects/bagon/ilanaveh/code/face.evoLVe/model', # the root to buffer your checkpoints
        LOG_ROOT = f'/home/projects/bagon/ilanaveh/code/face.evoLVe/log/IR_SE_50_ArcFace_Blur{cfg_blurs[11]}_try_not_open_out', # the root to log your train/val status
        BACKBONE_RESUME_ROOT = f'/home/projects/bagon/ilanaveh/code/face.evoLVe/model/IR_SE_50_ArcFace_Blur{cfg_blurs[11]}_try_not_open_out', # the root to resume training from a saved checkpoint
        HEAD_RESUME_ROOT = f'/home/projects/bagon/ilanaveh/code/face.evoLVe/model/IR_SE_50_ArcFace_Blur{cfg_blurs[11]}_try_not_open_out', # the root to resume training from a saved checkpoint

        BACKBONE_NAME = 'IR_SE_50', # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
        HEAD_NAME = 'ArcFace', # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
        LOSS_NAME = 'Focal', # support: ['Focal', 'Softmax']

        INPUT_SIZE = [112, 112], # support: [112, 112] and [224, 224]
        RGB_MEAN = [0.5, 0.5, 0.5], # for normalize inputs to [-1, 1]
        RGB_STD = [0.5, 0.5, 0.5],
        EMBEDDING_SIZE = 512, # feature dimension
        BATCH_SIZE = 512,
        DROP_LAST = True, # whether drop the last batch to ensure consistent batch_norm statistics
        LR = 0.1, # initial LR
        NUM_EPOCH = 120, # total epoch number (use the firt 1/25 epochs to warm up)  *-- change from 125
        WEIGHT_DECAY = 5e-4, # do not apply to batch_norm parameters
        MOMENTUM = 0.9,
        STAGES = [30, 60, 90], # epoch stages to decay learning rate  *-- Change from [35, 65, 95]

        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        MULTI_GPU = True, # flag to use multiple GPUs; if you choose to train with single GPU, you should first run "export CUDA_VISILE_DEVICES=device_id" to specify the GPU card you want to use
        GPU_ID = [0, 1, 2, 3], # specify your GPU ids  *-- Original: [0, 1, 2, 3]; for debugging - change to [0]
        PIN_MEMORY = True,
        NUM_WORKERS = 0,
        BLUR = cfg_blurs[4],
    ),

    12: dict(
        SEED = 1337, # random seed for reproduce results

        DATA_ROOT = '/home/projects/bagon/ilanaveh/data', # the parent root where your train/val/test data are stored
        MODEL_ROOT = '/home/projects/bagon/ilanaveh/code/face.evoLVe/model', # the root to buffer your checkpoints
        LOG_ROOT = f'/home/projects/bagon/ilanaveh/code/face.evoLVe/log/IR_SE_50_ArcFace_Blur{cfg_blurs[12]}_2workers', # the root to log your train/val status
        BACKBONE_RESUME_ROOT = f'/home/projects/bagon/ilanaveh/code/face.evoLVe/model/IR_SE_50_ArcFace_Blur{cfg_blurs[12]}_2workers', # the root to resume training from a saved checkpoint
        HEAD_RESUME_ROOT = f'/home/projects/bagon/ilanaveh/code/face.evoLVe/model/IR_SE_50_ArcFace_Blur{cfg_blurs[12]}_2workers', # the root to resume training from a saved checkpoint

        BACKBONE_NAME = 'IR_SE_50', # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
        HEAD_NAME = 'ArcFace', # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
        LOSS_NAME = 'Focal', # support: ['Focal', 'Softmax']

        INPUT_SIZE = [112, 112], # support: [112, 112] and [224, 224]
        RGB_MEAN = [0.5, 0.5, 0.5], # for normalize inputs to [-1, 1]
        RGB_STD = [0.5, 0.5, 0.5],
        EMBEDDING_SIZE = 512, # feature dimension
        BATCH_SIZE = 512,
        DROP_LAST = True, # whether drop the last batch to ensure consistent batch_norm statistics
        LR = 0.1, # initial LR
        NUM_EPOCH = 120, # total epoch number (use the firt 1/25 epochs to warm up)  *-- change from 125
        WEIGHT_DECAY = 5e-4, # do not apply to batch_norm parameters
        MOMENTUM = 0.9,
        STAGES = [30, 60, 90], # epoch stages to decay learning rate  *-- Change from [35, 65, 95]

        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        MULTI_GPU = True, # flag to use multiple GPUs; if you choose to train with single GPU, you should first run "export CUDA_VISILE_DEVICES=device_id" to specify the GPU card you want to use
        GPU_ID = [0, 1, 2, 3], # specify your GPU ids  *-- Original: [0, 1, 2, 3]; for debugging - change to [0]
        PIN_MEMORY = True,
        NUM_WORKERS = 2,
        BLUR = cfg_blurs[12],
    ),

    13: dict(
        SEED = 1337, # random seed for reproduce results

        DATA_ROOT = '/home/projects/bagon/ilanaveh/data', # the parent root where your train/val/test data are stored
        MODEL_ROOT = '/home/projects/bagon/ilanaveh/code/face.evoLVe/model', # the root to buffer your checkpoints
        LOG_ROOT = f'/home/projects/bagon/ilanaveh/code/face.evoLVe/log/IR_SE_50_ArcFace_Blur{cfg_blurs[13]}_norm_rgb', # the root to log your train/val status
        BACKBONE_RESUME_ROOT = f'/home/projects/bagon/ilanaveh/code/face.evoLVe/model/IR_SE_50_ArcFace_Blur{cfg_blurs[13]}_norm_rgb', # the root to resume training from a saved checkpoint
        HEAD_RESUME_ROOT = f'/home/projects/bagon/ilanaveh/code/face.evoLVe/model/IR_SE_50_ArcFace_Blur{cfg_blurs[13]}_norm_rgb', # the root to resume training from a saved checkpoint

        BACKBONE_NAME = 'IR_SE_50', # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
        HEAD_NAME = 'ArcFace', # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
        LOSS_NAME = 'Focal', # support: ['Focal', 'Softmax']

        INPUT_SIZE = [112, 112], # support: [112, 112] and [224, 224]
        RGB_MEAN =  [0.541, 0.432, 0.379], # Obtained from get_dataset_stats.
        RGB_STD = [0.286, 0.254, 0.248],
        EMBEDDING_SIZE = 512, # feature dimension
        BATCH_SIZE = 512,
        DROP_LAST = True, # whether drop the last batch to ensure consistent batch_norm statistics
        LR = 0.1, # initial LR
        NUM_EPOCH = 120, # total epoch number (use the firt 1/25 epochs to warm up)  *-- change from 125
        WEIGHT_DECAY = 5e-4, # do not apply to batch_norm parameters
        MOMENTUM = 0.9,
        STAGES = [30, 60, 90], # epoch stages to decay learning rate  *-- Change from [35, 65, 95]

        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        MULTI_GPU = True, # flag to use multiple GPUs; if you choose to train with single GPU, you should first run "export CUDA_VISILE_DEVICES=device_id" to specify the GPU card you want to use
        GPU_ID = [0, 1, 2, 3], # specify your GPU ids  *-- Original: [0, 1, 2, 3]; for debugging - change to [0]
        PIN_MEMORY = True,
        NUM_WORKERS = 0,
        BLUR = cfg_blurs[13],
    ),

    14: dict(
        SEED = 1337, # random seed for reproduce results

        DATA_ROOT = '/home/projects/bagon/ilanaveh/data', # the parent root where your train/val/test data are stored
        MODEL_ROOT = '/home/projects/bagon/ilanaveh/code/face.evoLVe/model', # the root to buffer your checkpoints
        LOG_ROOT = f'/home/projects/bagon/ilanaveh/code/face.evoLVe/log/IR_SE_50_ArcFace_Blur{cfg_blurs[14]}_norm_rgb_2workers', # the root to log your train/val status
        BACKBONE_RESUME_ROOT = f'/home/projects/bagon/ilanaveh/code/face.evoLVe/model/IR_SE_50_ArcFace_Blur{cfg_blurs[14]}_norm_rgb_2workers', # the root to resume training from a saved checkpoint
        HEAD_RESUME_ROOT = f'/home/projects/bagon/ilanaveh/code/face.evoLVe/model/IR_SE_50_ArcFace_Blur{cfg_blurs[14]}_norm_rgb_2workers', # the root to resume training from a saved checkpoint

        BACKBONE_NAME = 'IR_SE_50', # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
        HEAD_NAME = 'ArcFace', # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
        LOSS_NAME = 'Focal', # support: ['Focal', 'Softmax']

        INPUT_SIZE = [112, 112], # support: [112, 112] and [224, 224]
        RGB_MEAN =  [0.541, 0.432, 0.379], # Obtained from get_dataset_stats.
        RGB_STD = [0.286, 0.254, 0.248],
        EMBEDDING_SIZE = 512, # feature dimension
        BATCH_SIZE = 512,
        DROP_LAST = True, # whether drop the last batch to ensure consistent batch_norm statistics
        LR = 0.1, # initial LR
        NUM_EPOCH = 120, # total epoch number (use the firt 1/25 epochs to warm up)  *-- change from 125
        WEIGHT_DECAY = 5e-4, # do not apply to batch_norm parameters
        MOMENTUM = 0.9,
        STAGES = [30, 60, 90], # epoch stages to decay learning rate  *-- Change from [35, 65, 95]

        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        MULTI_GPU = True, # flag to use multiple GPUs; if you choose to train with single GPU, you should first run "export CUDA_VISILE_DEVICES=device_id" to specify the GPU card you want to use
        GPU_ID = [0, 1, 2, 3], # specify your GPU ids  *-- Original: [0, 1, 2, 3]; for debugging - change to [0]
        PIN_MEMORY = True,
        NUM_WORKERS = 2,
        BLUR = cfg_blurs[14],
    ),

    15: dict(
        SEED = 1337, # random seed for reproduce results

        DATA_ROOT = '/home/projects/bagon/ilanaveh/data', # the parent root where your train/val/test data are stored
        MODEL_ROOT = '/home/projects/bagon/ilanaveh/code/face.evoLVe/model', # the root to buffer your checkpoints
        LOG_ROOT = f'/home/projects/bagon/ilanaveh/code/face.evoLVe/log/IR_SE_50_ArcFace_Blur{cfg_blurs[15]}_4workers', # the root to log your train/val status
        BACKBONE_RESUME_ROOT = f'/home/projects/bagon/ilanaveh/code/face.evoLVe/model/IR_SE_50_ArcFace_Blur{cfg_blurs[15]}_4workers', # the root to resume training from a saved checkpoint
        HEAD_RESUME_ROOT = f'/home/projects/bagon/ilanaveh/code/face.evoLVe/model/IR_SE_50_ArcFace_Blur{cfg_blurs[15]}_4workers', # the root to resume training from a saved checkpoint

        BACKBONE_NAME = 'IR_SE_50', # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
        HEAD_NAME = 'ArcFace', # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
        LOSS_NAME = 'Focal', # support: ['Focal', 'Softmax']

        INPUT_SIZE = [112, 112], # support: [112, 112] and [224, 224]
        RGB_MEAN = [0.5, 0.5, 0.5], # for normalize inputs to [-1, 1]
        RGB_STD = [0.5, 0.5, 0.5],
        EMBEDDING_SIZE = 512, # feature dimension
        BATCH_SIZE = 512,
        DROP_LAST = True, # whether drop the last batch to ensure consistent batch_norm statistics
        LR = 0.1, # initial LR
        NUM_EPOCH = 120, # total epoch number (use the firt 1/25 epochs to warm up)  *-- change from 125
        WEIGHT_DECAY = 5e-4, # do not apply to batch_norm parameters
        MOMENTUM = 0.9,
        STAGES = [30, 60, 90], # epoch stages to decay learning rate  *-- Change from [35, 65, 95]

        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        MULTI_GPU = True, # flag to use multiple GPUs; if you choose to train with single GPU, you should first run "export CUDA_VISILE_DEVICES=device_id" to specify the GPU card you want to use
        GPU_ID = [0, 1, 2, 3], # specify your GPU ids  *-- Original: [0, 1, 2, 3]; for debugging - change to [0]
        PIN_MEMORY = True,
        NUM_WORKERS = 4,
        BLUR = cfg_blurs[15],
    ),

    16: dict(
        SEED = 1337, # random seed for reproduce results

        DATA_ROOT = '/home/projects/bagon/ilanaveh/data', # the parent root where your train/val/test data are stored
        MODEL_ROOT = '/home/projects/bagon/ilanaveh/code/face.evoLVe/model', # the root to buffer your checkpoints
        LOG_ROOT = f'/home/projects/bagon/ilanaveh/code/face.evoLVe/log/IR_SE_50_ArcFace_Blur{cfg_blurs[16]}_2wrkrs_1gpu', # the root to log your train/val status
        BACKBONE_RESUME_ROOT = f'/home/projects/bagon/ilanaveh/code/face.evoLVe/model/IR_SE_50_ArcFace_Blur{cfg_blurs[16]}_2wrkrs_1gpu', # the root to resume training from a saved checkpoint
        HEAD_RESUME_ROOT = f'/home/projects/bagon/ilanaveh/code/face.evoLVe/model/IR_SE_50_ArcFace_Blur{cfg_blurs[16]}_2wrkrs_1gpu', # the root to resume training from a saved checkpoint

        BACKBONE_NAME = 'IR_SE_50', # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
        HEAD_NAME = 'ArcFace', # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
        LOSS_NAME = 'Focal', # support: ['Focal', 'Softmax']

        INPUT_SIZE = [112, 112], # support: [112, 112] and [224, 224]
        RGB_MEAN = [0.5, 0.5, 0.5], # for normalize inputs to [-1, 1]
        RGB_STD = [0.5, 0.5, 0.5],
        EMBEDDING_SIZE = 512, # feature dimension
        BATCH_SIZE = 512,
        DROP_LAST = True, # whether drop the last batch to ensure consistent batch_norm statistics
        LR = 0.1, # initial LR
        NUM_EPOCH = 120, # total epoch number (use the firt 1/25 epochs to warm up)  *-- change from 125
        WEIGHT_DECAY = 5e-4, # do not apply to batch_norm parameters
        MOMENTUM = 0.9,
        STAGES = [30, 60, 90], # epoch stages to decay learning rate  *-- Change from [35, 65, 95]

        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        MULTI_GPU = True, # flag to use multiple GPUs; if you choose to train with single GPU, you should first run "export CUDA_VISILE_DEVICES=device_id" to specify the GPU card you want to use
        GPU_ID = [0], # specify your GPU ids  *-- Original: [0, 1, 2, 3]; for debugging / single gpu - change to [0]
        PIN_MEMORY = True,
        NUM_WORKERS = 2,
        BLUR = cfg_blurs[16],
    ),

    17: dict(
        SEED = 1337, # random seed for reproduce results

        DATA_ROOT = '/home/projects/bagon/ilanaveh/data', # the parent root where your train/val/test data are stored
        MODEL_ROOT = '/home/projects/bagon/ilanaveh/code/face.evoLVe/model', # the root to buffer your checkpoints
        LOG_ROOT = f'/home/projects/bagon/ilanaveh/code/face.evoLVe/log/IR_SE_50_ArcFace_Blur{cfg_blurs[17]}_2workers', # the root to log your train/val status
        BACKBONE_RESUME_ROOT = f'/home/projects/bagon/ilanaveh/code/face.evoLVe/model/IR_SE_50_ArcFace_Blur{cfg_blurs[17]}_2workers', # the root to resume training from a saved checkpoint
        HEAD_RESUME_ROOT = f'/home/projects/bagon/ilanaveh/code/face.evoLVe/model/IR_SE_50_ArcFace_Blur{cfg_blurs[17]}_2workers', # the root to resume training from a saved checkpoint

        BACKBONE_NAME = 'IR_SE_50', # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
        HEAD_NAME = 'ArcFace', # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
        LOSS_NAME = 'Focal', # support: ['Focal', 'Softmax']

        INPUT_SIZE = [112, 112], # support: [112, 112] and [224, 224]
        RGB_MEAN = [0.5, 0.5, 0.5], # for normalize inputs to [-1, 1]
        RGB_STD = [0.5, 0.5, 0.5],
        EMBEDDING_SIZE = 512, # feature dimension
        BATCH_SIZE = 512,
        DROP_LAST = True, # whether drop the last batch to ensure consistent batch_norm statistics
        LR = 0.1, # initial LR
        NUM_EPOCH = 120, # total epoch number (use the firt 1/25 epochs to warm up)  *-- change from 125
        WEIGHT_DECAY = 5e-4, # do not apply to batch_norm parameters
        MOMENTUM = 0.9,
        STAGES = [30, 60, 90], # epoch stages to decay learning rate  *-- Change from [35, 65, 95]

        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        MULTI_GPU = True, # flag to use multiple GPUs; if you choose to train with single GPU, you should first run "export CUDA_VISILE_DEVICES=device_id" to specify the GPU card you want to use
        GPU_ID = [0, 1, 2, 3], # specify your GPU ids  *-- Original: [0, 1, 2, 3]; for debugging - change to [0]
        PIN_MEMORY = True,
        NUM_WORKERS = 2,
        BLUR = cfg_blurs[17],
    ),

    18: dict(
        SEED = 1337, # random seed for reproduce results

        DATA_ROOT = '/home/projects/bagon/ilanaveh/data', # the parent root where your train/val/test data are stored
        MODEL_ROOT = '/home/projects/bagon/ilanaveh/code/face.evoLVe/model', # the root to buffer your checkpoints
        LOG_ROOT = f'/home/projects/bagon/ilanaveh/code/face.evoLVe/log/IR_SE_50_ArcFace_Blur{cfg_blurs[18]}_2workers', # the root to log your train/val status
        BACKBONE_RESUME_ROOT = f'/home/projects/bagon/ilanaveh/code/face.evoLVe/model/IR_SE_50_ArcFace_Blur{cfg_blurs[18]}_2workers', # the root to resume training from a saved checkpoint
        HEAD_RESUME_ROOT = f'/home/projects/bagon/ilanaveh/code/face.evoLVe/model/IR_SE_50_ArcFace_Blur{cfg_blurs[18]}_2workers', # the root to resume training from a saved checkpoint

        BACKBONE_NAME = 'IR_SE_50', # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
        HEAD_NAME = 'ArcFace', # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
        LOSS_NAME = 'Focal', # support: ['Focal', 'Softmax']

        INPUT_SIZE = [112, 112], # support: [112, 112] and [224, 224]
        RGB_MEAN = [0.5, 0.5, 0.5], # for normalize inputs to [-1, 1]
        RGB_STD = [0.5, 0.5, 0.5],
        EMBEDDING_SIZE = 512, # feature dimension
        BATCH_SIZE = 512,
        DROP_LAST = True, # whether drop the last batch to ensure consistent batch_norm statistics
        LR = 0.1, # initial LR
        NUM_EPOCH = 120, # total epoch number (use the firt 1/25 epochs to warm up)  *-- change from 125
        WEIGHT_DECAY = 5e-4, # do not apply to batch_norm parameters
        MOMENTUM = 0.9,
        STAGES = [30, 60, 90], # epoch stages to decay learning rate  *-- Change from [35, 65, 95]

        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        MULTI_GPU = True, # flag to use multiple GPUs; if you choose to train with single GPU, you should first run "export CUDA_VISILE_DEVICES=device_id" to specify the GPU card you want to use
        GPU_ID = [0, 1, 2, 3], # specify your GPU ids  *-- Original: [0, 1, 2, 3]; for debugging - change to [0]
        PIN_MEMORY = True,
        NUM_WORKERS = 2,
        BLUR = cfg_blurs[18],
    ),

    19: dict(
        SEED = 1337, # random seed for reproduce results

        DATA_ROOT = '/home/projects/bagon/ilanaveh/data', # the parent root where your train/val/test data are stored
        MODEL_ROOT = '/home/projects/bagon/ilanaveh/code/face.evoLVe/model', # the root to buffer your checkpoints
        LOG_ROOT = f'/home/projects/bagon/ilanaveh/code/face.evoLVe/log/IR_SE_50_ArcFace_Blur{cfg_blurs[19]}_2workers', # the root to log your train/val status
        BACKBONE_RESUME_ROOT = f'/home/projects/bagon/ilanaveh/code/face.evoLVe/model/IR_SE_50_ArcFace_Blur{cfg_blurs[19]}_2workers', # the root to resume training from a saved checkpoint
        HEAD_RESUME_ROOT = f'/home/projects/bagon/ilanaveh/code/face.evoLVe/model/IR_SE_50_ArcFace_Blur{cfg_blurs[19]}_2workers', # the root to resume training from a saved checkpoint

        BACKBONE_NAME = 'IR_SE_50', # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
        HEAD_NAME = 'ArcFace', # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
        LOSS_NAME = 'Focal', # support: ['Focal', 'Softmax']

        INPUT_SIZE = [112, 112], # support: [112, 112] and [224, 224]
        RGB_MEAN = [0.5, 0.5, 0.5], # for normalize inputs to [-1, 1]
        RGB_STD = [0.5, 0.5, 0.5],
        EMBEDDING_SIZE = 512, # feature dimension
        BATCH_SIZE = 512,
        DROP_LAST = True, # whether drop the last batch to ensure consistent batch_norm statistics
        LR = 0.1, # initial LR
        NUM_EPOCH = 120, # total epoch number (use the firt 1/25 epochs to warm up)  *-- change from 125
        WEIGHT_DECAY = 5e-4, # do not apply to batch_norm parameters
        MOMENTUM = 0.9,
        STAGES = [30, 60, 90], # epoch stages to decay learning rate  *-- Change from [35, 65, 95]

        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        MULTI_GPU = True, # flag to use multiple GPUs; if you choose to train with single GPU, you should first run "export CUDA_VISILE_DEVICES=device_id" to specify the GPU card you want to use
        GPU_ID = [0, 1, 2, 3], # specify your GPU ids  *-- Original: [0, 1, 2, 3]; for debugging - change to [0]
        PIN_MEMORY = True,
        NUM_WORKERS = 2,
        BLUR = cfg_blurs[19],
    )

}
