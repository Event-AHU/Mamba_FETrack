# Mamba_FETrack V2

* **Mamba-FETrack V2: Revisiting State Space Model for Frame-Event based Visual Object Tracking**, Shiao Wang, Ju Huang, Qingchuan Ma, Jinfeng Gao, Chunyi Xu, Xiao Wang, Lan Chen, Bo Jiang
  [[Paper]()]
  [[Code](https://github.com/Event-AHU/Mamba_FETrack/tree/main/Mamba_FETrackV2)]
  

###  :dart: Abstract 
Combining traditional RGB cameras with bio-inspired event cameras for robust object tracking has garnered increasing attention in recent years. However, most existing multimodal tracking algorithms depend heavily on high-complexity Vision Transformer architectures for feature extraction and fusion across modalities. This not only leads to substantial computational overhead but also limits the effectiveness of cross-modal interactions. In this paper, we propose an efficient RGB-Event object tracking framework based on the linear-complexity Vision Mamba network, termed Mamba-FETrack V2. Specifically, we first design a lightweight Prompt Generator that utilizes embedded features from each modality, together with a shared prompt pool, to dynamically generate modality-specific learnable prompt vectors. These prompts, along with the modality-specific embedded features, are then fed into a Vision Mamba-based FEMamba backbone, which facilitates prompt-guided feature extraction, cross-modal interaction, and fusion in a unified manner. Finally, the fused representations are passed to the tracking head for accurate target localization. Extensive experimental evaluations on multiple RGB-Event tracking benchmarks, including short-term COESOT dataset and long-term datasets, i.e., FE108 and FELT V2, demonstrate the superior performance and efficiency of the proposed tracking framework.

### Mamba-FETrack V2 Framework 
<p align="center">
<img src="https://github.com/Event-AHU/Mamba_FETrack/blob/main/Mamba_FETrackV2/figures/MambaFETrack_framework_V3.jpg" alt="framework" width="700"/>
</p>


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






