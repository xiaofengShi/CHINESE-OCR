conda create -n chinese-ocr python=2.7 pip scipy numpy PIL jupyter##运用conda 创建python环境
source activate chinese-ocr
pip install easydict -i https://pypi.tuna.tsinghua.edu.cn/simple/ ##选择国内源，速度更快
pip install keras==2.0.8  -i https://pypi.tuna.tsinghua.edu.cn/simple/  
pip install Cython opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple/ 
pip install matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple/ 
pip install -U pillow -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install  h5py lmdb mahotas -i https://pypi.tuna.tsinghua.edu.cn/simple/
conda install pytorch=0.1.12 torchvision -c soumith
conda install tensorflow=1.3 ##解决cuda报错相关问题
cd ./ctpn/lib/utils
sh make-for-cpu.sh


