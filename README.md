# 本文基于tensorflow、keras/pytorch实现对自然场景的文字检测及端到端的OCR中文文字识别

## 参考github仓库
[TOTAL](https://github.com/chineseocr/chinese-ocr/tree/chinese-ocr-python-3.6)-挂掉了

[CRNN—pytorch](https://github.com/meijieru/crnn.pytorch.git)


# 实现功能

- 文字方向检测 0、90、180、270度检测 
- 文字检测 后期将切换到keras版本文本检测 实现keras端到端的文本检测及识别
- 不定长OCR识别


## 环境部署
``` 
Bash
##GPU环境
sh setup.sh
##CPU环境
sh setup-cpu.sh
##CPU python3环境
sh setup-python3.sh

使用环境：python3.6+tensorflow1.7+cpu/gpu
```

# 模型训练
* 一共分为3个网络
* 	**1. 文本方向检测网络-Classify(vgg16)**
*  **2. 文本区域检测网络-CTPN(CNN+RNN)**
*  **3. EndToEnd文本识别网络-CRNN(CNN+GRU/LSTM+CTC)**

# 文字方向检测-vgg分类
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
- 此外参考了了tensorflow版本的资源仓库：[TF:LSTM-CTC_loss](https://github.com/xiaofengShi/CTC_TF) 

# 这个仓库咋用呢
## 如果你只是测试一下
```
运行demo.py  写入测试图片的路径即可，如果想要显示ctpn的结果，修改文件./ctpn/ctpn/other.py 的draw_boxes函数的最后部分，cv2.inwrite('dest_path',img)，如此，可以得到ctpn检测的文字区域框以及图像的ocr识别结果
```
## 如果你想训练这个网络
### 1 对ctpn进行训练
* 定位到路径--./ctpn/ctpn/train_net.py
* 预训练的vgg网络路径[VGG_imagenet.npy](https://pan.baidu.com/s/1JO_ZojA5bkmJZsnxsShgkg)
将预训练权重下载下来，pretrained_model指向该路径即可,
此外整个模型的预训练权重[checkpoint](https://pan.baidu.com/s/1aT-vHgq7nvLy4M_T6SwR1Q)
* ctpn数据集[还是百度云](https://pan.baidu.com/s/1NXFmdP_OgRF42xfHXUhBHQ)
数据集下载完成并解压后，将.ctpn/lib/datasets/pascal_voc.py 文件中的pascal_voc 类中的参数self.devkit_path指向数据集的路径即可

### 2 对crnn进行训练
* keras版本 ./train/keras_train/train_batch.py  model_path--指向预训练权重位置 
MODEL_PATH---指向模型训练保存的位置
[keras模型预训练权重](https://pan.baidu.com/s/1vTG6-i_bFMWxQ_7xF06usg)
* pythorch版本./train/pytorch-train/crnn_main.py
```
parser.add_argument(
    '--crnn',
    help="path to crnn (to continue training)",
    default=预训练权重的路径，看你下载的预训练权重在哪啦)
parser.add_argument(
    '--experiment',
    help='Where to store samples and models',
    default=模型训练的权重保存位置,这个自己指定)
```
[pytorch预训练权重](https://pan.baidu.com/s/1LEDNHEr3luloB7eZK6GOeA)




# 识别结果展示
## 文字检测及OCR识别结果
![ctpn原始图像1](./test/ttttt.png)
`===========================================================`
![ctpn检测1](./test/test1.png)
`===========================================================`
![ctpn+crnn结果1](./test/ttttt_result.png)

主要是因为训练的时候，只包含中文和英文字母，因此很多公式结构是识别不出来的
### 看看纯文字的
![ctpn原始图像2](./test/test.png)
`===========================================================`
![ctpn检测2](./test/test_pre.png)
`===========================================================`
![ctpn+crnn结果2](./test/test_result.png)

# 未完待续
### tensorflow版本crnn，计划尝试当前的各种trick(dropuout,bn,learning_decay等)
```
可以看到，对于纯文字的识别结果还是阔以的呢，感觉可以在crnn网络在加以改进，现在的crnn中的cnn有点浅，
并且rnn层为单层双向+attention，目前正在针对这个地方进行改动，使用迁移学习，以restnet为特征提取层，
使用多层双向动态rnn+attention+ctc的机制，将模型加深，目前正在进行模型搭建，结果好的话就发上来，不好的话只能凉凉了~~~~
```





## 参考

- [pytorch 实现crnn](https://github.com/meijieru/crnn.pytorch.git)    
- [keras-crnn 版本实现参考](https://www.zhihu.com/question/59645822)  
- [tensorflow-crnn](https://github.com/ilovin/lstm_ctc_ocr)
- [tensorflow-ctpn](https://github.com/eragonruan/text-detection-ctpn )
- [CAFFE-CTPN](https://github.com/tianzhi0549/CTPN)

