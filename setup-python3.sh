source activate base
#conda create -n chinese-ocr3 python=3.6 pip scipy numpy Pillow jupyter
#source activate chinese-ocr3
pip install easydict -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install keras  -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install Cython opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install -U pillow -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install  h5py lmdb mahotas -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install futures==3.1.1 -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install tensorflow -i https://pypi.tuna.tsinghua.edu.cn/simple/
cd ./ctpn/lib/utils
./make-for-cpu.sh
