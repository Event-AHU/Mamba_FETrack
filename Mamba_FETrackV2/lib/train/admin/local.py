class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/wangx/DATA/Code/wangshiao/MambaFETrackV2/Mamba_FETrackV2_submit'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir + '/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = self.workspace_dir + '/pretrained_models'
        # self.coesot_dir = "/wangx/DATA/Dataset/COESOT/train/"
        # self.coesot_val_dir = self.coesot_dir
        # self.fe108_dir = '/wangx/DATA/Dataset/fe108/train/'
        # self.fe108_val_dir = self.fe108_dir
        self.felt_dir = '/wangx/DATA/Dataset/FELT/train/'
        self.felt_val_dir = self.felt_dir