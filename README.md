# tf-faster-rcnn-cpu
Xinlei Chen's tf-faster-rcnn work with cpu.
and changed by kidtic . 

## CPU运行方法
tf-faster-rcnn-cpu支持cpu和gpu运行,原来需要更改3处地方的文件来切换cpu使用，现在只需要更改一处地方即可：
1. 将lib/model/config.py中 GPU_USE_CONFIG_DF设置成False


然后编译lib库即可

## 安装方法
1.克隆
```buildoutcfg 
git clone https://github.com/kidtic/tf-faster-rcnn-cpu.git
```
2.修改你的gpu架构
```
cd tf-faster-rcnn/lib
# Change the GPU architecture (-arch) if necessary
vim setup.py
```
3.编译lib
```
make clean
make
cd ..
```
4.运行cocoAPI
```
./cocoAPI.sh
```

## 运行demo并测试预训练模型
1. [下载](https://pan.baidu.com/s/1lcTYvckpk_nsj2H2JECVrw)模型
2. 将模型放到文件夹```/output/res101 ```下
3. 运行demo
```
GPU_ID=0
CUDA_VISIBLE_DEVICES=${GPU_ID} ./tools/demo.py
```