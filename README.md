# Welcome to braincraft_img_classifier
This repo is for classifying the image between three classes such as person, group of person and focus object class. 
# Installing

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
Finally, put the detectron2 git repo in the braincraft_img_classifier folder by below cmd.

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
There are about 3500 annotated text samples. Below are a couple of samples: <br/>

<br/>
However, according to our chosen model, we have to discipline teh original text. To some extent, we were performed a thorough  investigate among the datasets for checking the irregular annotated samples
<br/>
Eventually, we analyzed and measured the annotated labels quality and necessities. The original datasets have nearly 20 different 
labels and among them some are unnecssesary and we figured out that these are irrelevant to our job. Thus, we eradicated them from the sample annotations. We chose 7 different labels for our model.
<br/>
For the final datasets, we filtered around 3467 ideal samples. Below are a few examples of our 10 ideal samples:  
<br/> <br/>
<p align="center">
  <img src="https://github.com/h-muhammed/hisab_ner/blob/feature/develop/imgs/datasets.PNG" title="Ideal Samples">
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
  <img src="https://github.com/h-muhammed/hisab_ner/blob/feature/develop/imgs/loss.png" title="Ideal Samples">
 </p>

#### Inference  <br/> 
Put inference text in `src/datasets/pred_text.txt`  <br/> Download the checkpoints model from the [shared link](https://drive.google.com/drive/folders/102B6IUpwJ-hj659a5elTQUeboSOpzrLe?usp=sharing)  and put it in the `/output/checkpoints/` folder  <br/>
```
python predict.py --modle_name BanglaBert --gpu_ids -1
```
#### Inference  <br/>
<p align="center">
  <img src="https://github.com/h-muhammed/hisab_ner/blob/feature/develop/imgs/infer.PNG" title="Ideal Samples">
 </p>

# Whats Next i.e: To-Do  <br/>

- [ ] Design and develop an web API for demonstrating the prediction result. <br/>
    - [ ] tools: fastapi <br/>
- [ ] Dockerize the environment for independent platforms.  <br/>
    - [ ] tools: docker, docker-compose <br/>

# Acknowledgement
Specail thanks goes to Hisab coding test system for assinging and sharing a well organized resource and clear instructions. <br/> <br/>
[paper](https://arxiv.org/abs/2205.00034)  <br/>
[Bengali_Ner](https://github.com/Rifat1493/Bengali-NER)  <br/>
[towardsdatascience blo](https://towardsdatascience.com/named-entity-recognition-with-bert-in-pytorch-a454405e0b6a)
