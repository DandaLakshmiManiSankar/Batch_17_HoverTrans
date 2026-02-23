# Hover-Trans: Anatomy-aware HoVer-Transformer for ROI-free Breast Cancer Diagnosis in Ultrasound Images
7
## Introduction
The implementation of: <br>
[**Hover-Trans: Anatomy-aware HoVer-Transformer for ROI-free Breast Cancer Diagnosis in Ultrasound Images**](https://ieeexplore.ieee.org/document/10015121)
## Requirements
- python 3.9
- Pytorch 1.10.1
- torchvision 0.11.2
- opencv-python
- pandas
- scipy
## Setup
### Installation
Clone the repo and install required packages:
```
git clone https://github.com/yuhaomo/HoVerTrans.git
cd HoVerTrans
pip install -r requirements.txt
```
### Dataset
-  You can download our dataset ([GDPH&SYSUCC](https://1drv.ms/u/s!AgOtqK2ZncKlgoxsmt-UYbEwMyZY2g?e=INNhyK)) and unpack them into the ./data folder.
```
./data
└─GDPH&SYSUCC
      ├─label.csv
      └─img
          ├─benign(0).png
          ├─benign(1).png
          ├─benign(2).png
          ├─malignant(0).png
          ├─malignant(1).png
          ...
```
- The format of the label.csv is as follows:
```
+------------------+-------+
| name             | label |
+------------------+-------+
| benign(0).png    |   0   |
| benign(1).png    |   0   |
| benign(2).png    |   0   |
| malignant(0).png |   1   |
| malignant(1).png |   1   |
...
```
### Training
```
python train.py --data_path ./data/GDPH&SYSUCC/img --csv_path ./data/GDPH&SYSUCC/label.csv --batch_size 32 --class_num 2 --epochs 250 --lr 0.0001 
```
