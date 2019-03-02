##   1. Install CUDA Toolkit 9.0

Wow, why should I install 9.0 and not 9.1 or 9.2? Because Tensorflow 1.8 is built with 9.0, so you can just download the prebuild version. If you really need the bleeding edge not-stable version  you can waste few days and compile it.

From the CUDA Toolkit Archive download version specific for your OS. In case of Ubuntu don’t use run-file installer — it destroys graphics driver and you will be stuck in the login loop. Use deb remote or local.

Download from here: https://developer.nvidia.com/cuda-90-download-archive

##  2. Install CUDNN 7.0.5 for CUDA 9.0

CUDNN provides functions specific to neural networks that are used by Tensorflow. It is extremely important to download the correct version for your CUDA and OS.

You will need to register. Registration is free.

https://developer.nvidia.com/rdp/cudnn-archive

##  3. Install python

If you are already using python just go to next step. There are many ways to install python. 

Native python
You will need to install python and package management system pip. Here is simple way to install python3 on Ubuntu.
```
sudo apt-get update
sudo apt-get install -y build-essential libssl-dev libffi-dev python3-dev python3 python3-pip
```
##  4. Install Tensorflow

When you prepared your python environment, you are ready to install Tensorflow:


Install Tensorflow with GPU support:
```
pip3 install tensorflow-gpu
```
Install Tensorflow without GPU support:
```
pip3 install tensorflow
```
##  5. Download Object Detection repository

If you don’t have git installed, install git with following command:

```
sudo apt install -y git
```

Now we will clone Tensorflow models repositories:

```
git clone https://github.com/tensorflow/models
```

##  6. Install Object Detection API

Now we will install the APU using this guide:https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md

So let’s install necessary libraries:
```
sudo apt-get install protobuf-compiler python-pil python-lxml python-tk
pip3 install --user Cython
pip3 install --user contextlib2
pip3 install --user jupyter
pip3 install --user matplotlib
```

##  7. Install COCO API

```
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
cp -r pycocotools /models/research/
```

##  8. Install Protoc
See https://gist.github.com/sofyanhadia/37787e5ed098c97919b8c593f0ec44d8

After downloading run following command:
```
# From /models/research/
protoc object_detection/protos/*.proto --python_out=.
```

##  9. Build and Install research

```
python3 setup.py build
python3 setup.py install
```


##  10. Install openCV

```
https://pypi.org/project/opencv-python/
```

##  11. Install numPy

```
python3 -m pip install numpy

```

##  12. Change path object detection in source code
```
PATH_OBJECT_DETECTION = "<full path>/models/research/object_detection/"
```

##  13. Download model

See list model and download
```
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
```


Uzip where located source code

OR

For lazy

Uncomment
```
# ## Download Model

# opener = urllib.request.URLopener()
# opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
# tar_file = tarfile.open(MODEL_FILE)
# for file in tar_file.getmembers():
#   file_name = os.path.basename(file.name)
#   if 'frozen_inference_graph.pb' in file_name:
#     tar_file.extract(file, os.getcwd())

```


change MODEL_NAME

##  14. Start

```
python3 main_withou_gpu.py
```

