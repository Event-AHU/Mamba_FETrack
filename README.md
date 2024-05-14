# Mamba_FETrack

* **Mamba-FETrack: Frame-Event Tracking via State Space Model**, Ju Huang, Shiao Wang, Shuai Wang, Zhe Wu, Xiao Wang, Bo Jiang
  [[Paper](https://arxiv.org/abs/2404.18174)]
  [[Code](https://github.com/Event-AHU/Mamba_FETrack)]


### News 


# :dart: Abstract 
RGB-Event based tracking is an emerging research topic, focusing on how to effectively integrate heterogeneous multi-modal data (synchronized exposure video frames and asynchronous pulse Event stream). Existing works typically employ Transformer based networks to handle these modalities and achieve decent accuracy through input-level or feature-level fusion on multiple datasets. However, these trackers require significant memory consumption and computational complexity due to the use of self-attention mechanism. This paper proposes a novel RGB-Event tracking framework, Mamba-FETrack, based on the State Space Model (SSM) to achieve high-performance tracking while effectively reducing computational costs and realizing more efficient tracking. Specifically, we adopt two modality-specific Mamba backbone networks to extract the features of RGB frames and Event streams. Then, we also propose to boost the interactive learning between the RGB and Event features using the Mamba network. The fused features will be fed into the tracking head for target object localization. Extensive experiments on FELT and FE108 datasets fully validated the efficiency and effectiveness of our proposed tracker. Specifically, our Mamba-based tracker achieves 43.5/55.6 on the SR/PR metric, while the ViT-S based tracker (OSTrack) obtains 40.0/50.9. The GPU memory cost of ours and ViT-S based tracker is 13.98GB and 15.44GB, which decreased about $9.5\%$. The FLOPs and parameters of ours/ViT-S based OSTrack are 59GB/1076GB and 7MB/60MB, which decreased about $94.5\%$ and $88.3\%$, respectively. We hope this work can bring some new insights to the tracking field and greatly promote the application of the Mamba architecture in tracking. 

### FETrack Framework 
<p align="center">
<img src="https://github.com/Event-AHU/Mamba_FETrack/blob/main/figures/Mamba_track_framework.jpg" alt="framework" width="700"/>
</p>




### Environment Settings 
```
conda create -n mamba_fetrack python=3.10.13
conda activate mamba_fetrack
```
First, install the package for Vim

### Dataset Download 



### Training and Testing 




### Evaluation Toolkit 




### Experimental Results 



### Acknowledgment 
[[OSTrack](https://github.com/botaoye/OSTrack)] 
[[Mamba](https://github.com/state-spaces/mamba)] 
[[FELT](https://github.com/Event-AHU/FELT_SOT_Benchmark)] 
[[CEUTrack](https://github.com/Event-AHU/COESOT)] 
[[FE108](https://zhangjiqing.com/dataset/contact)] 

### :newspaper: Citation 
If you think this survey is helpful, please feel free to leave a star ⭐️ and cite our paper:
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





