from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    # settings.coesot_path = ''
    # settings.fe108_path = ''
    settings.felt_path = '/wangx/DATA/Dataset/FELT/'
    
    settings.network_path = ''    # Where tracking networks are stored.
    settings.prj_dir = '/wangx/DATA/Code/wangshiao/MambaFETrackV2/Mamba_FETrackV2_submit'
    settings.result_plot_path = settings.prj_dir + '/output/test/result_plots'
    settings.results_path = settings.prj_dir + '/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = settings.prj_dir + '/output'

    return settings

