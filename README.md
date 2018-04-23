# `本文基于tensorflow、keras/pytorch实现对自然场景的文字检测及端到端的OCR中文文字识别`

# 参考github仓库
[TOTAL](https://github.com/chineseocr/chinese-ocr/tree/chinese-ocr-python-3.6)-挂掉了

[CRNN—pytorch](https://github.com/meijieru/crnn.pytorch.git)


# 实现功能

- 文字方向检测 0、90、180、270度检测 
- 文字检测 后期将切换到keras版本文本检测 实现keras端到端的文本检测及识别
- 不定长OCR识别


## 环境部署
``` Bash
##GPU环境
sh setup.sh
##CPU环境
sh setup-cpu.sh
##CPU python3环境
sh setup-python3.sh
```

# 模型训练
* 一共分为3个网络
* 	**1. 文本方向检测网络-Classify(vgg16)**
*  **2. 文本区域检测网络-CTPN(CNN+RNN)**
*  **3. EndToEnd文本识别网络-CRNN(CNN+GRU/LSTM+CTC)**

# 文字方向检测
```
基于图像分类，在VGG16模型的基础上，训练0、90、180、270度检测的分类模型.
详细代码参考angle/predict.py文件，训练图片8000张，准确率88.23%
```
模型地址[BaiduCloud](https://pan.baidu.com/s/1zquQNdO0MUsLMsuwxbgPYg)

# 文字区域检测CTPN
支持CPU、GPU环境，一键部署，
[文本检测训练参考](https://github.com/eragonruan/text-detection-ctpn)  
 

# OCR 端到端识别:CRNN
## ocr识别采用GRU+CTC端到到识别技术，实现不分隔识别不定长文字
提供keras 与pytorch版本的训练代码，在理解keras的基础上，可以切换到pytorch版本，此版本更稳定
- 此外还添加了tensorflow版本的资源仓库：[TF:LSTM-CTC_loss](https://github.com/ilovin/lstm_ctc_ocr) 
## 训练keras版本的crnn   

``` Bash
cd train & sh train-keras.sh   
```

## 训练pytorch版本的crnn   

``` Bash
cd train & sh train-pytorch.sh   
```
   
# 识别结果展示
## 文字检测及OCR识别结果
<div>
<img width="300" height="300" src="https://github.com/chineseocr/chinses-ocr/blob/master/img/tmp.jpg"/>
<img width="300" height="300" src="https://github.com/chineseocr/chinses-ocr/blob/master/img/tmp.png"/>
</div>

### 倾斜文字 

<div>
<img width="300" height="300" src="https://github.com/chineseocr/chinses-ocr/blob/master/img/tmp1.jpg"/>
<img width="300" height="300" src="https://github.com/chineseocr/chinses-ocr/blob/master/img/tmp1.png"/>
</div>

## 参考

- [pytorch 实现crnn](https://github.com/meijieru/crnn.pytorch.git)    
- [keras-crnn 版本实现参考](https://www.zhihu.com/question/59645822)  
- [tensorflow-crnn](https://github.com/ilovin/lstm_ctc_ocr)
- [tensorflow-ctpn](https://github.com/eragonruan/text-detection-ctpn 
)
- [CAFFE-CTPN](https://github.com/tianzhi0549/CTPN   
)

