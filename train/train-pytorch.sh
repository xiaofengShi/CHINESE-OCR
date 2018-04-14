cd pytorch-train
nohup python crnn_main.py --cuda --adadelta >/tmp/crnnlog10.log 2>&1 &
