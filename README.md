# Welcome to braincraft_img_classifier
This repo is for classifying the image between three classes such as person, group of person and focus object class. 
# Environment creation

This project is primarily developed in Linux. <br/>
### For Linux: <br />
Create a virtual environment by below cmd <br />
```
python -m venv braincraft
```
For activation <br />
```
source braincraft/bin/activate
```
First install the python 3.10 version by below cmd.

```
sudo apt update
sudo apt-get install python3.10
```
And then install relevant dependencies by below cmd <br />

```
pip install -r requirement.txt
```
Finally, since we utilized the detecrton2 library from Meta ai research for implementing maskrcnn and faster-rcnn 
then we need to put the detectron2 git repo in the braincraft_img_classifier folder by below cmd.

```
git clone https://github.com/facebookresearch/detectron2.git
```
Then install the detectron2 by below cmd.
```
python -m pip install -e detectron2
```
### For Windows: <br />
Create a virtual environment by below cmd <br />
```
pip install virtualenv
virtualenv braincraft
```
For activation <br />
```
braincraft\Scripts\activate
``` 
And then install relevant dependencies by below cmd <br />

```
pip install -r requirement.txt
```
Finally, put the detectron2 git repo in the braincraft_img_classifier folder by below cmd.

```
git clone https://github.com/facebookresearch/detectron2.git
```
Then install the detectron2 by below cmd.
```
python -m pip install -e detectron2
```


# Datasets pipelining
There are about 500 annotated image samples training and 100 samples for evaluating the performance. Below are a couple of samples: <br/>
 
<br/> <br/>
<p align="center">
  <img src="" title="Ideal Samples">
 </p>


# Project Structure
    
    braincraft_img_classifier
        |___config
        |___datasets
        |___detectron2
        |___log
        |___prediction
        |___evaluate.py
        |___train.py
        |___utils.py
        |___visualizer_withPredict.py
        |___requirements.txt
        |___README.md


# Implementation


#### Train <br />
For Maskrcnn pretrain model
For gpu, use --device `cuda` else `cpu` <br/>
```
python3 train.py -e "config/R101_FPN_ep3.yaml" --device cuda 
```
For faster-rcnn pretrain model
For gpu, use --device `cuda` else `cpu` <br/>
```
python3 train.py -e "config/faster_rcnn_R_50_C4_3x_Ep1.yaml" --device cuda 
```

#### Accuracy and Loss graph 
<p align="center">
  <img src="" title="Ideal Samples">
 </p>

#### Inference  <br/> 

Put the `model_final.pth` file in the `log/R101-FPN_3x_MAXiter50000/` folder
For maskrcnn and cpu
```
python3 visualizer_withPredict.py -e "config/R101_FPN_ep3.yaml" --device cpu
```
For faster-rcnn and cpu
```
python3 visualizer_withPredict.py -e "config/faster_rcnn_R_50_C4_3x_Ep1.yaml" --device cpu
```

#### Inference  <br/>
<p align="center">
  <img src="" title="Ideal Samples">
 </p>

# Whats Next i.e: To-Do  <br/>

- [ ] Design and develop an web API for demonstrating the prediction result. <br/>
    - [ ] tools: fastapi <br/>
- [ ] Dockerize the environment for independent platforms.  <br/>
    - [ ] tools: docker, docker-compose <br/>

# Acknowledgement
Specail thanks goes to Braincraft recruitment team for assigning and sharing a well-organized and clear instructions. <br/> <br/>
[maskrcnn](https://arxiv.org/abs/1703.06870)  <br/>
[fasterrcnn](https://arxiv.org/abs/1703.06870)  <br/>

