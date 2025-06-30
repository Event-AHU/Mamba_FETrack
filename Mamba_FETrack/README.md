# Mamba_FETrack

* **Mamba-FETrack: Frame-Event Tracking via State Space Model**, Ju Huang, Shiao Wang, Shuai Wang, Zhe Wu, Xiao Wang, Bo Jiang
  [[Paper](https://arxiv.org/abs/2404.18174)]
  [[Code](https://github.com/Event-AHU/Mamba_FETrack)]


### News 


* [2024.06.25] This work is accepted by **PRCV-2024** [[第七届中国模式识别与计算机视觉大会 (The 7th Chinese Conference on Pattern Recognition and Computer Vision PRCV 2024)](http://2024.prcv.cn/)]. 
* [2024.05.15] Source code and weights are all released.
* [2024.04.28] The arXiv paper is released [[arXiv](https://arxiv.org/abs/2404.18174)] 


###  :dart: Abstract 
RGB-Event based tracking is an emerging research topic, focusing on how to effectively integrate heterogeneous multi-modal data (synchronized exposure video frames and asynchronous pulse Event stream). Existing works typically employ Transformer based networks to handle these modalities and achieve decent accuracy through input-level or feature-level fusion on multiple datasets. However, these trackers require significant memory consumption and computational complexity due to the use of self-attention mechanism. This paper proposes a novel RGB-Event tracking framework, Mamba-FETrack, based on the State Space Model (SSM) to achieve high-performance tracking while effectively reducing computational costs and realizing more efficient tracking. Specifically, we adopt two modality-specific Mamba backbone networks to extract the features of RGB frames and Event streams. Then, we also propose to boost the interactive learning between the RGB and Event features using the Mamba network. The fused features will be fed into the tracking head for target object localization. Extensive experiments on FELT and FE108 datasets fully validated the efficiency and effectiveness of our proposed tracker. Specifically, our Mamba-based tracker achieves 43.5/55.6 on the SR/PR metric, while the ViT-S based tracker (OSTrack) obtains 40.0/50.9. The GPU memory cost of ours and ViT-S based tracker is 13.98GB and 15.44GB, which decreased about $9.5\%$. The FLOPs and parameters of ours/ViT-S based OSTrack are 59GB/1076GB and 7MB/60MB, which decreased about $94.5\%$ and $88.3\%$, respectively. We hope this work can bring some new insights to the tracking field and greatly promote the application of the Mamba architecture in tracking. 




### Mamba-FETrack Framework 
<p align="center">
<img src="https://github.com/Event-AHU/Mamba_FETrack/blob/main/Mamba_FETrack/figures/Mamba_track_framework.jpg" alt="framework" width="700"/>
</p>



### Download Checkpoint  
  Download [pre-trained](https://pan.baidu.com/s/1-5q4hK2LWj16K6R2PHSdPw?pwd=AHUT) and put it under `$/pretrained_models`.



### Download Trained Weights for Model 
Download the trained model weights from [Mamba_FETrack_ep0050.pth](https://pan.baidu.com/s/1avb4gcWJmS2YIkmzjDCKcg?pwd=AHUT) and put it under `$/output/checkpoints/train/mamba_fetrack/mamba_fetrack_felt` for test directly.



### Training and Testing Script 
```
# train
python tracking/train.py --script mamba_fetrack --config mamba_fetrack_felt --save_dir ./output --mode single --nproc_per_node 1 --use_wandb 0
python tracking/train.py --script mamba_fetrack --config mamba_fetrack_fe108 --save_dir ./output --mode single --nproc_per_node 1 --use_wandb 0

# test
python tracking/test.py mamba_fetrack mamba_fetrack_felt --dataset felt --threads 1 --num_gpus 1
python tracking/test.py mamba_fetrack mamba_fetrack_fe108 --dataset fe108 --threads 1 --num_gpus 1
```



### Evaluation Toolkit 
 * **Evaluation methods on the FELT dataset**
1. Download the FELT_eval_toolkit from [FELT_eval_toolkit (Passcode：AHUT)](https://pan.baidu.com/s/1jZkKQpwP-mSTMnZYO79Z9g?pwd=AHUT), and open it with Matlab (over Matlab R2020).
2. add your [tracking results (Passcode：AHUT)](https://pan.baidu.com/s/13nCvBezuaBQP5hZzftEEIw?pwd=AHUT) in `$/felt_tracking_results/` and modify the name in `$/utils/config_tracker.m`
3. run `Evaluate_FELT_benchmark_SP_PR_only.m` for the overall performance evaluation, including AUC, PR, NPR.
4. run `plot_BOC.m` for BOC score evaluation and figure plot.
5. run `plot_radar.m` for attributes radar figrue plot.
6. run `Evaluate_FELT_benchmark_attributes.m` for attributes analysis and figure saved in `$/res_fig/`. 
 * **Evaluation methods on the FE108 dataset**
1. add your [tracking results (Passcode：AHUT)](https://pan.baidu.com/s/1xZypplOReASeK38GQAzUKg?pwd=AHUT) in `$/output/test/tracking_results/mamba_fetrack/`
2. run `$/tracking/analysis_results.py` for the overall performance evaluation, including AUC, PR, NPR.


### Visualization 

* The code of visualization can be found at: **Mamba_FETrack/lib/test/tracker/show_CAM.py** 

<p align="center">
<img src="https://github.com/Event-AHU/Mamba_FETrack/blob/main/Mamba_FETrack/figures/activationMAPs.jpg" alt="framework" width="700"/>
</p>



### Experimental Results 
* **Experimental results (AUC/PR) on FE108 dataset**
<p align="center">
<img src="https://github.com/Event-AHU/Mamba_FETrack/blob/main/Mamba_FETrack/figures/FE108.jpg" alt="framework" width="700"/>
</p>

* **Experimental results (SR/PR) on FELT dataset**
<p align="center">
<img src="https://github.com/Event-AHU/Mamba_FETrack/blob/main/Mamba_FETrack/figures/FELT.png" alt="framework" width="700"/>
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
@misc{huang2024mambafetrack,
      title={Mamba-FETrack: Frame-Event Tracking via State Space Model}, 
      author={Ju Huang and Shiao Wang and Shuai Wang and Zhe Wu and Xiao Wang and Bo Jiang},
      year={2024},
      eprint={2404.18174},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```






