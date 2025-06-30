# Mamba_FETrack V2

* **Mamba-FETrack V2: Revisiting State Space Model for Frame-Event based Visual Object Tracking**, Shiao Wang, Ju Huang, Qingchuan Ma, Jinfeng Gao, Chunyi Xu, Xiao Wang, Lan Chen, Bo Jiang
  [[Paper]()]
  [[Code](https://github.com/Event-AHU/Mamba_FETrack/tree/main/Mamba_FETrackV2)]


### News 



###  :dart: Abstract 
Combining traditional RGB cameras with bio-inspired event cameras for robust object tracking has garnered increasing attention in recent years. However, most existing multimodal tracking algorithms depend heavily on high-complexity Vision Transformer architectures for feature extraction and fusion across modalities. This not only leads to substantial computational overhead but also limits the effectiveness of cross-modal interactions. In this paper, we propose an efficient RGB-Event object tracking framework based on the linear-complexity Vision Mamba network, termed Mamba-FETrack V2. Specifically, we first design a lightweight Prompt Generator that utilizes embedded features from each modality, together with a shared prompt pool, to dynamically generate modality-specific learnable prompt vectors. These prompts, along with the modality-specific embedded features, are then fed into a Vision Mamba-based FEMamba backbone, which facilitates prompt-guided feature extraction, cross-modal interaction, and fusion in a unified manner. Finally, the fused representations are passed to the tracking head for accurate target localization. Extensive experimental evaluations on multiple RGB-Event tracking benchmarks, including short-term COESOT dataset and long-term datasets, i.e., FE108 and FELT V2, demonstrate the superior performance and efficiency of the proposed tracking framework.

### Mamba-FETrack V2 Framework 
<p align="center">
<img src="https://github.com/Event-AHU/Mamba_FETrack/blob/main/Mamba_FETrackV2/figures/MambaFETrack_framework_V3.jpg" alt="framework" width="700"/>
</p>

### Environment Settings 
* **Install environment using conda**
```
conda create -n fetrackv2 python=3.10.13
conda activate fetrackv2
```

 * **Install the package for Vim**
```
conda install cudatoolkit==11.8 -c nvidia
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" cuda-nvcc
conda install packaging
pip install -r vim_requirements.txt
```
* **Install the mamba-1.1.1 and casual-conv1d-1.1.3 for mamba**

Download the [mamba-1.1.1](https://github.com/state-spaces/mamba/releases/download/v1.1.1/mamba_ssm-1.1.1+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl) and [source code](https://github.com/state-spaces/mamba/archive/refs/tags/v1.1.1.zip) and place it in the project path of Mamba_FETrackV2. Go to source code and install the corresponding environment.
```
cd mamba-1.1.1
pip install .
```

Download the [casual-conv1d-1.1.3](https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.1.3/causal_conv1d-1.1.3+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl) and [source code](https://github.com/Dao-AILab/causal-conv1d/archive/refs/tags/v1.1.3.zip) and place it in the project path of Mamba_FETrackV2.  Go to source code and install the corresponding environment.
```
cd ..
cd causal-conv1d-1.1.3
pip install .
```
    
* **Install the package for tracking**
```
bash install.sh
```

* **Run the following command to set paths for this project**
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```

* **After running this command, you can also modify paths by editing these two files**
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

### Download Dataset  
  Download tracking datasets [FELT V2]() OR [FE108](https://zhangjiqing.com/dataset/), OR [COESOT](https://pan.baidu.com/s/12XDlKABlz3lDkJJEDvsu9A?pwd=AHUT ) and put it in `./data`.



### Download pretrained models  
  Download [pre-trained](https://drive.google.com/drive/folders/1sTHPc0fCNrGZi-xS1OpuYU9KWT9aJNef?hl=zh-cn) and put it under `$/pretrained_models`.



### Download Trained Weights for Model 
Download the trained model weights from [Mamba_FETrack_ep0050.pth](https://drive.google.com/drive/folders/1ihlhp4_gIHmXpIkrJ2wTfleja_wA-Pqa?hl=zh-cn) and put it under `$/output/checkpoints/train/mamba_fetrack/mamba_fetrack_felt` for test FELT V2 dataset directly.



### Training and Testing Script 
```
# train
python tracking/train.py --script mamba_fetrack --config mamba_fetrack_felt --save_dir ./output --mode single --nproc_per_node 1 --use_wandb 0
python tracking/train.py --script mamba_fetrack --config mamba_fetrack_coesot --save_dir ./output --mode single --nproc_per_node 1 --use_wandb 0
python tracking/train.py --script mamba_fetrack --config mamba_fetrack_fe108 --save_dir ./output --mode single --nproc_per_node 1 --use_wandb 0

# test
python tracking/test.py mamba_fetrack mamba_fetrack_felt --dataset felt --threads 1 --num_gpus 1
python tracking/test.py mamba_fetrack mamba_fetrack_coesot --dataset coesot --threads 1 --num_gpus 1
python tracking/test.py mamba_fetrack mamba_fetrack_fe108 --dataset fe108 --threads 1 --num_gpus 1
```



### Evaluation Toolkit 
 * **Evaluation methods on the FELT dataset**
1. Download the FELT_eval_toolkit from [FELT_eval_toolkit (Passcode：AHUT)](https://pan.baidu.com/s/1jZkKQpwP-mSTMnZYO79Z9g?pwd=AHUT), and open it with Matlab.
3. add your [tracking results (Passcode：AHUE)](https://pan.baidu.com/s/1i9ye9QM-EeZRzRpJ7R8p1g?pwd=AHUE) in `$/felt_tracking_results/` and modify the name in `$/utils/config_tracker.m`
4. run `Evaluate_FELT_benchmark_SP_PR_only.m` for the overall performance evaluation, including AUC, PR, NPR.

 * **Evaluation methods on the COESOT dataset**
1. Download the FELT_eval_toolkit from [COESOT_eval_toolkit ([Passcode：AHUT](https://github.com/Event-AHU/COESOT/tree/main/COESOT_eval_toolkit))], and open it with Matlab.
2. add your [tracking results (Passcode：AHUE)](https://pan.baidu.com/s/1i9ye9QM-EeZRzRpJ7R8p1g?pwd=AHUE) in `$/coesot_tracking_results/` and modify the name in `$/utils/config_tracker.m`
3. run `Evaluate_FELT_benchmark_SP_PR_only.m` for the overall performance evaluation, including AUC, PR, NPR.

 * **Evaluation methods on the FE108 dataset**
1. add your [tracking results (Passcode：AHUE)](https://pan.baidu.com/s/1i9ye9QM-EeZRzRpJ7R8p1g?pwd=AHUE) in `$/output/test/tracking_results/mamba_fetrack/`
2. run `$/tracking/analysis_results.py` for the overall performance evaluation, including AUC, PR.


### Visualization 

* The code of visualization can be found at: **Mamba_FETrack/lib/test/tracker/show_CAM.py** 

<p align="center">
<img src="https://github.com/Event-AHU/Mamba_FETrack/blob/main/Mamba_FETrackV2/figures/response_map.jpg" alt="framework" width="700"/>
</p>



### Experimental Results 
* **Experimental results on FE108 dataset**
<p align="center">
<img src="https://github.com/Event-AHU/Mamba_FETrack/blob/main/Mamba_FETrackV2/figures/fe108_result.png" alt="framework" width="700"/>
</p>

* **Experimental results on COESOT dataset and FELT V2 dataset**
<p align="center">
<img src="https://github.com/Event-AHU/Mamba_FETrack/blob/main/Mamba_FETrackV2/figures/feltv2%26coesot.png" alt="framework" width="700"/>
</p>

### Acknowledgment 
[[OSTrack](https://github.com/botaoye/OSTrack)] 
[[Mamba](https://github.com/state-spaces/mamba)] 
[[FELT](https://github.com/Event-AHU/FELT_SOT_Benchmark)] 
[[CEUTrack](https://github.com/Event-AHU/COESOT)] 
[[FE108](https://zhangjiqing.com/dataset/contact)] 



### :newspaper: Citation 
If you think this paper is helpful, please feel free to leave a star ⭐️ and cite our paper:
```bibtex
```






