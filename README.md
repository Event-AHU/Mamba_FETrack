# Mamba_FETrack



# :collision: News 
* [2024.06.25] This work is accepted by **PRCV-2024** [[第七届中国模式识别与计算机视觉大会 (The 7th Chinese Conference on Pattern Recognition and Computer Vision PRCV 2024)](http://2024.prcv.cn/)]. 
* [2024.05.15] Source code and weights are all released.
* [2024.04.28] The arXiv paper is released [[arXiv](https://arxiv.org/abs/2404.18174)] 


# :hammer: Environment Settings 
* **Install environment using conda**
```
conda create -n mamba_fetrack python=3.10.13
conda activate mamba_fetrack
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

Download the [mamba-1.1.1](https://github.com/state-spaces/mamba/releases/download/v1.1.1/mamba_ssm-1.1.1+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl) and [source code](https://github.com/state-spaces/mamba/archive/refs/tags/v1.1.1.zip) and place it in the project path. Go to source code and install the corresponding environment.
```
cd mamba-1.1.1
pip install .
```

Download the [casual-conv1d-1.1.3](https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.1.3/causal_conv1d-1.1.3+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl) and [source code](https://github.com/Dao-AILab/causal-conv1d/archive/refs/tags/v1.1.3.zip) and place it in the project path.  Go to source code and install the corresponding environment.
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

# :dvd: Download Dataset  
Download tracking datasets [FELT V2]() OR [FE108](https://zhangjiqing.com/dataset/), OR [COESOT](https://pan.baidu.com/s/12XDlKABlz3lDkJJEDvsu9A?pwd=AHUT ) and put it in `./data`.




# :cupid: Acknowledgment 
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





