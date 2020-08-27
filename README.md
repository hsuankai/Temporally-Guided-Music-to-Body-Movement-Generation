# Temporally-Guided-Music-to-Body-Movement-Generation
This is the pytorch implementation of music-to-body-movement model described in the paper:  

>[Hsuan-Kai Kao](https://github.com/hsuankai) and [Li Su](https://www.iis.sinica.edu.tw/pages/lisu/index_en.html). Temporally Guided Music-to-Body-Movement Generation, ACM Multimedia 2020  
This project is part of the [Automatic Music Concert Animation (AMCA)](https://sites.google.com/view/mctl/research/automatic-music-concert-animation) project of the Institute of Information Science, Academia Sinica, Taiwan.

![image](https://github.com/hsuankai/Temporally-Guided-Music-to-Body-Movement-Generation/blob/master/demo_animation.gif)
![image](https://github.com/hsuankai/Temporally-Guided-Music-to-Body-Movement-Generation/blob/master/demo_skeleton.gif)

## Quick start
To get started as quickly as possible, follow the instructions in this section. This should allow you train a model from scratch and inference your own violin music.

### Dependencies
- Python 3+ distribution
- Pytorch >= 1.0.1, CUDA 10.0  
- Install requirements by running: `pip install -r requirement.txt`
- To visualize predictions, install ffmpeg by running: `apt-get install ffmpeg`

### Data
In the paper, we use 14-fold cross validation for 14 musical pieces in evaluation. However, to test model performance for simplicity, we here only provide trainging data and test data for one fold, and all data are already preprocessed in feature level. You can download **train.pkl** and **test.pkl** automatically by executing **train.py** and **test.py** or use **data.py**.

We also provide [`URMP.txt`](https://github.com/hsuankai/Temporally-Guided-Music-to-Body-Movement-Generation/blob/master/data/URMP.txt) which list the musical pieces used in our cross-dataset evaluation.

### Training from scratch
To reproduce the results, run following commands:
```
python train.py --d_model 512 --gpu_ids 0
python test.py --plot_path xxx.mp4 --output_path xxx.pkl
```
If you have problem with limited gpu memory usage, try to decrease `--d_model` or use multi-gpu `--gpu_ids 0,1,2`.
- `--plot_path` make video of predicted playing movement. We here specify one of violinist for visualization.
- `--output_path` save predicted keypoints and ground truth, whose dimensions is N x K x C, where N is the number of frames, K is the number of keypoints and C is three axes x, y and z.

### Inference in the wild
If you want to make video and get predicted keypoints for custom audio data by pretrained model, run following commands:
```
python inference.py --inference_audio xxx.wav --plot_path xxx.mp4 --output_path xxx.pkl
```
`--plot_path` and `--output_path` are the same as described in **test.py**, and you need to put the path of your violin music on argument `--inference_audio`.
