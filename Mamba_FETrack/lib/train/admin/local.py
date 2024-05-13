class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/huangju/Mamba_FETrack'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/huangju/Mamba_FETrack/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/huangju/Mamba_FETrack/pretrained_networks'
        self.coesot_dir = '/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/dataset/COESOT/train/'
        self.coesot_val_dir = '/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/dataset/COESOT/test'
        self.fe108_dir = '/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/dataset/fe108/train'
        self.fe108_val_dir = '/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/dataset/fe108/train'
        self.visevent_dir = '/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/dataset/VisEvent/train'
        self.visevent_val_dir = '/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/dataset/VisEvent/train'
        self.felt_dir = '/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/dataset/FELT/FELT_train'
        self.lasher_dir = '/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/dataset/LasHeR/LasHeR_Divided_TraningSet&TestingSet/TrainingSet/trainingset'
        self.depthtrack_dir = '/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/dataset/DepthTrack/train'