## YOLOv8 + MobileSAM在AirBox上实现自动标注

本文档将介绍如何在AirBox部署YOLOv8 + MobileSAM程序，实现自动标注（图片或者视频）的功能。

###  1、连接AirBox

将AirBox通过WAN口与计算机相连，然后设置计算机端的IP地址。

以windows操作系统为例，打开`控制面板\网络和 Internet\网络和共享中心`，点击`以太网2——>属性`，手动设置IP地址为192.168.150.2，如图所示。连接成功后，AirBox的IP即是192.168.150.1。

![image-20231017142503889](assets/1.png)


然后把网线插到LAN口AirBox就可以接入互联网了。

###  2、环境配置

打开ssh远程连接工具。以mobaxterm为例：点击`会话——>SSH`，远程主机填192.168.150.1，其他选项默认，点击”OK“。

![image-20231017143107426](assets/2.png)

用户名：linaro

密码：linaro

然后安装SDK，在[算能官网](https://developer.sophgo.com/site/index/material/37/all.html)下载SDK-23.05.01：

![690e63de1007e0ba6ba03a33eb73261](assets/3.png)

下载好之后，请到`Release_230501-public.zip\Release_230501-public\sophon-sail_20230605_085400\sophon-sail_3.5.0.tar.gz\sophon-sail_3.5.0.tar\sophon-sail\python_wheels\soc\libsophon-0.4_sophonmw-0.6\py38\`路径下提取`sophon_arm-3.5.0-py3-none-any.whl`文件到AirBox上。

将whl文件拷贝到AirBox的/data下，输入`pip3 install "/data/sophon_arm-3.5.0-py3-none-any.whl"`安装SDK。


###  3、文件下载

GitHub链接如下：https://github.com/ZillaRU/AnnoSeg

将项目文件拷贝到AirBox的/data下。

项目结构树如下：

```
.
├── README.md
├── app.py
├── argparse3-0.0.0-py3-none-any.whl
├── assets
│   ├── image.png
│   └── image1.png
├── requirements.txt
├── sam
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-38.pyc
│   │   └── sam_demo_fast.cpython-38.pyc
│   ├── sam_demo_fast.py
│   └── sam_utils
│       ├── __init__.py
│       ├── __pycache__
│       │   ├── __init__.cpython-38.pyc
│       │   └── transforms.cpython-38.pyc
│       └── transforms.py
├── test_imgs
│   ├── cat.jpg
│   ├── picture1.jpg
│   └── picture3.jpg
├── test_videos
│   ├── 10s_Trim.mp4
│   └── dog_trim.mp4
├── utils
│   ├── __init__,py
│   ├── __pycache__
│   │   ├── tools.cpython-38.pyc
│   │   └── tools_gradio.cpython-38.pyc
│   ├── tools.py
│   └── tools_gradio.py
├── weight
│   ├── mask_decoder.bmodel
│   ├── mobilesam_encoder_hwc.bmodel
│   ├── no_mask_embed_weight.npz
│   ├── not_a_point_embed_weight.npz
│   ├── point_embeddings.npz
│   ├── positional_encoding_gaussian_matrix.npy
│   └── sam_tinyvit_0915.bmodel
└── yolov8
    ├── BM1684X_models
    │   ├── yolov8s_fp16_1b.bmodel
    │   ├── yolov8s_fp32_1b.bmodel
    │   ├── yolov8s_int8_1b.bmodel
    │   └── yolov8s_int8_4b.bmodel
    ├── __init__.py
    ├── __pycache__
    │   ├── __init__.cpython-38.pyc
    │   ├── postprocess_numpy.cpython-38.pyc
    │   ├── utils.cpython-38.pyc
    │   └── yolov8_det_demo.cpython-38.pyc
    ├── postprocess_numpy.py
    ├── utils.py
    └── yolov8_det_demo.py

13 directories, 43 files
```
###  4、运行程序

安装依赖：`pip3 install -r requirements.txt`，运行`python3 app.py`，如果有包没安装pip安装即可，运行成功截图如下：

![image-20231017151550017](assets/4.png)

这里我的端口是7860，在浏览器中`192.168.150.1:7860`即可愉快的玩耍啦~

# 标注图片
上传图片，选取需要标注的类别（可多选），点击Annotate，标注结果显示在右侧。
![image annotation](assets/image.png)

# 标注视频
上传视频，选取需要标注的类别（单选），点击Annotate，标注完成的视频在右侧可下载。
![video annotation](assets/image1.png)
