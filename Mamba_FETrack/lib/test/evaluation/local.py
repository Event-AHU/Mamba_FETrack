from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.coesot_path = '/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/dataset/COESOT'
    settings.fe108_path = '/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/dataset/FE108'
    settings.network_path = '/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/huangju/Mamba_FETrack/output/test/networks'    # Where tracking networks are stored.
    settings.prj_dir = '/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/huangju/Mamba_FETrack'
    settings.result_plot_path = '/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/huangju/Mamba_FETrack/output/test/result_plots'
    settings.results_path = '/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/huangju/Mamba_FETrack/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/huangju/Mamba_FETrack/output'
    settings.segmentation_path = '/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/huangju/Mamba_FETrack/output/test/segmentation_results'
    settings.visevent_path = '/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/dataset/VisEvent'
    settings.felt_path = '/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/dataset/FELT'
    settings.lasher_path = '/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/dataset/LasHeR/LasHeR_Divided_TraningSet&TestingSet/TestingSet'
    settings.rgbt234_path = '/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/dataset/RGBT234'

    return settings

