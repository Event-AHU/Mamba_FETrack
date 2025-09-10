# 🔎 Mamba-FETrack V2

* **Mamba-FETrack V2: Revisiting State Space Model for Frame-Event based Visual Object Tracking**, Shiao Wang, Ju Huang, Qingchuan Ma, Jinfeng Gao, Chunyi Xu, Xiao Wang, Lan Chen, Bo Jiang
  [[Paper](https://arxiv.org/abs/2506.23783)]
  [[Code](https://github.com/Event-AHU/Mamba_FETrack/tree/main/Mamba_FETrackV2)]
  

#  :dart: Abstract 
Combining traditional RGB cameras with bio-inspired event cameras for robust object tracking has garnered increasing attention in recent years. However, most existing multimodal tracking algorithms depend heavily on high-complexity Vision Transformer architectures for feature extraction and fusion across modalities. This not only leads to substantial computational overhead but also limits the effectiveness of cross-modal interactions. In this paper, we propose an efficient RGB-Event object tracking framework based on the linear-complexity Vision Mamba network, termed Mamba-FETrack V2. Specifically, we first design a lightweight Prompt Generator that utilizes embedded features from each modality, together with a shared prompt pool, to dynamically generate modality-specific learnable prompt vectors. These prompts, along with the modality-specific embedded features, are then fed into a Vision Mamba-based FEMamba backbone, which facilitates prompt-guided feature extraction, cross-modal interaction, and fusion in a unified manner. Finally, the fused representations are passed to the tracking head for accurate target localization. Extensive experimental evaluations on multiple RGB-Event tracking benchmarks, including short-term COESOT dataset and long-term datasets, i.e., FE108 and FELT V2, demonstrate the superior performance and efficiency of the proposed tracking framework.

### Mamba-FETrack V2 Framework 
<p align="center">
<img src="https://github.com/Event-AHU/Mamba_FETrack/blob/main/Mamba_FETrackV2/figures/MambaFETrack_framework_V3.jpg" alt="framework" width="850"/>
</p>


### Download pretrained models  
  Download [pre-trained](https://pan.baidu.com/s/1_xIb24UUz0uUxiXACALNUw?pwd=AHUE) and put it under `$/pretrained_models`.



### Download Trained Weights for Model 
Download the trained model weights from [Mamba_FETrack_ep0050.pth (Passcode：AHUE)](https://pan.baidu.com/s/1CzFwj4W4HXv3NUzb619-Yw?pwd=AHUE) and put it under `$/output/checkpoints/train/mamba_fetrack/mamba_fetrack_felt` for test directly.



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
# :dvd: Download Dataset  
Download tracking datasets [FELT V2](https://pan.baidu.com/s/1AiUTsvvsCKj8lWuc-821Eg?pwd=ahut) OR [FE108](https://zhangjiqing.com/dataset/), OR [COESOT](https://pan.baidu.com/s/12XDlKABlz3lDkJJEDvsu9A?pwd=AHUT ) and put it in `./data`.


# :triangular_ruler: Evaluation Toolkit 
 * **Evaluation methods on the FELT dataset**
1. Download the FELT_eval_toolkit from [FELT_eval_toolkit (Passcode：AHUT)](https://pan.baidu.com/s/1jZkKQpwP-mSTMnZYO79Z9g?pwd=AHUT), and open it with Matlab.
3. add your [tracking results (Passcode：AHUE)](https://pan.baidu.com/s/1i9ye9QM-EeZRzRpJ7R8p1g?pwd=AHUE) in `$/felt_tracking_results/` and modify the name in `$/utils/config_tracker.m`
4. run `Evaluate_FELT_benchmark_SP_PR_only.m` for the overall performance evaluation, including AUC, PR, NPR.

 * **Evaluation methods on the COESOT dataset**
1. Download the COESOT_eval_toolkit from [COESOT_eval_toolkit ([Passcode：AHUT](https://github.com/Event-AHU/COESOT/tree/main/COESOT_eval_toolkit))], and open it with Matlab.
2. add your [tracking results (Passcode：AHUE)](https://pan.baidu.com/s/1i9ye9QM-EeZRzRpJ7R8p1g?pwd=AHUE) in `$/coesot_tracking_results/` and modify the name in `$/utils/config_tracker.m`
3. run `Evaluate_FELT_benchmark_SP_PR_only.m` for the overall performance evaluation, including AUC, PR, NPR.

 * **Evaluation methods on the FE108 dataset**
1. add your [tracking results (Passcode：AHUE)](https://pan.baidu.com/s/1i9ye9QM-EeZRzRpJ7R8p1g?pwd=AHUE) in `$/output/test/tracking_results/mamba_fetrack/`
2. run `$/tracking/analysis_results.py` for the overall performance evaluation, including SR, PR.



# :video_camera: Visualization 

* The code of visualization can be found at: **/lib/test/tracker/show_CAM.py** 

<p align="center">
<img src="https://github.com/Event-AHU/Mamba_FETrack/blob/main/Mamba_FETrackV2/figures/response_map.jpg" alt="framework" width="700"/>
</p>



# :bookmark_tabs: Experimental Results 
* **Experimental results on FE108 dataset**
<p align="center">
<img src="https://github.com/Event-AHU/Mamba_FETrack/blob/main/Mamba_FETrackV2/figures/fe108_result.png" alt="framework" width="700"/>
</p>

* **Experimental results on COESOT dataset and FELT V2 dataset**
<p align="center">
<img src="https://github.com/Event-AHU/Mamba_FETrack/blob/main/Mamba_FETrackV2/figures/feltv2%26coesot.png" alt="framework" width="700"/>
</p>


# :newspaper: Citation 
If you think this paper is helpful, please feel free to leave a star ⭐️ and cite our paper:
```bibtex
@misc{wang2025mambafetrackv2revisitingstate,
      title={Mamba-FETrack V2: Revisiting State Space Model for Frame-Event based Visual Object Tracking}, 
      author={Shiao Wang and Ju Huang and Qingchuan Ma and Jinfeng Gao and Chunyi Xu and Xiao Wang and Lan Chen and Bo Jiang},
      year={2025},
      eprint={2506.23783},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```






