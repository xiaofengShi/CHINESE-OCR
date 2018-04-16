# 本文基于tensorflow、keras/pytorch实现对自然场景的文字检测及端到端的OCR中文文字识别

# 参考github仓库
[TOTAL](https://github.com/chineseocr/chinese-ocr/tree/chinese-ocr-python-3.6)
[CRNN—pytorch](https://github.com/meijieru/crnn.pytorch.git)


# 实现功能

- [x]  文字方向检测 0、90、180、270度检测 
- [x] 文字检测 后期将切换到keras版本文本检测 实现keras端到端的文本检测及识别
- [x] 不定长OCR识别
- [x] 增加python3.6 支持


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

## 训练keras版本的crnn   

``` Bash
cd train & sh train-keras.sh   
```

## 训练pytorch版本的crnn   

``` Bash
cd train & sh train-pytorch.sh   
```
# 文字方向检测
基于图像分类，在VGG16模型的基础上，训练0、90、180、270度检测的分类模型，详细代码参考angle/predict.py文件，训练图片8000张，准确率88.23%。
模型地址[百度云](https://pan.baidu.com/s/1pM2ha5P)下载

# 文字检测
支持CPU、GPU环境，一键部署，
[文本检测训练参考](https://github.com/eragonruan/text-detection-ctpn)  
 

# OCR 端到端识别:GRU+CTC
## ocr识别采用GRU+CTC端到到识别技术，实现不分隔识别不定长文字
提供keras 与pytorch版本的训练代码，在理解keras的基础上，可以切换到pytorch版本，此版本更稳定
- 此外还添加了tensorflow版本的资源仓库：[TF:LSTM-CTC_loss](https://github.com/ilovin/lstm_ctc_ocr)    


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
```
1.crnn 
https://github.com/meijieru/crnn.pytorch.git      
2.keras-crnn 版本实现参考 https://www.zhihu.com/question/59645822  

3.tensorflow-crnn 
https://github.com/ilovin/lstm_ctc_ocr      

3.ctpn
https://github.com/eragonruan/text-detection-ctpn 
https://github.com/tianzhi0549/CTPN   
```

