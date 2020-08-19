# Temporally-Guided-Music-to-Body-Movement-Generation
This is the implementation of music-to-body-movement model described in the paper:  

>[Hsuan-Kai Kao](https://github.com/hsuankai) and [Li Su](https://www.iis.sinica.edu.tw/pages/lisu/index_en.html). Temporally Guided Music-to-Body-Movement Generation.
In ACM Multimedia 2020
This project is part of the [Automatic Music Concert Animation (AMCA)](https://sites.google.com/view/mctl/research/automatic-music-concert-animation) project of the Institute of Information Science, Academia Sinica, Taiwan.

![image](https://github.com/hsuankai/Temporally-Guided-Music-to-Body-Movement-Generation/blob/master/demo_animation.gif)
![image](https://github.com/hsuankai/Temporally-Guided-Music-to-Body-Movement-Generation/blob/master/demo_skeleton.gif)

## Quick start
### Training from scratch
If you want to reproduce the results, run following commands:
```
python train.py 
python test.py
```
This will automatically download the data used in the paper and train a new model.
### Inference in the wild
If you want to generate skeleton data and plot animation by custom audio data, you can run following commands:
```
python inference.py --inference_data test.wav --animation_output test.mp4
```
