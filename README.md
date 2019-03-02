# Configuration

## 1.1 Install CUDA 10.0 + cuDNN 7.4.2 + NCCL 2.3.5

### 1.1.1 Install CUDA 10.0

Download [CUDA 10.0](https://developer.nvidia.com/cuda-10.0-download-archive?target_os=Linux "CUDA 10.0")

```
sudo dpkg -i cuda-repo-ubuntu1804-10-0-local-10.0.130-410.48_1.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu1804-10-0-local-10.0.130-410.48_1.0-1/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda
```

After installation is complete, add PATH variable by adding following line to the bashrc by running
```
nano ~/.bashrcnano ~/.bashrc
```

adding

```
export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}
```

at the end of file. Save and exit.

Check driver version and CUDA toolkit to ensure that everything went well
```
cat /proc/driver/nvidia/version
nvcc -V
```

### 1.2 Install cuDNN

Download [cuDNN 7.4.2](https://developer.nvidia.com/rdp/cudnn-archive#a-collapse742-10 "cuDNN 7.4.2")

Run following commands in he folders with deb files:
```
sudo dpkg -i libcudnn7_7.4.2.24-1+cuda10.0_amd64.deb
sudo dpkg -i libcudnn7-dev_7.4.2.24-1+cuda10.0_amd64.deb
sudo dpkg -i libcudnn7-doc_7.4.2.24-1+cuda10.0_amd64.deb
```

The installation is completed. Let’s verify it by following instructions or run:
```
cp -r /usr/src/cudnn_samples_v7/ $HOME
cd $HOME/cudnn_samples_v7/mnistCUDNN
make clean && make
./mnistCUDNN
```

If cuDNN was installed properly you will see a message: Test passed
Feel free to remove copied files from HOME/cudnn_samples_v7

### 1.3 Install NCCL 2.4.2

Download [NCCL](https://developer.nvidia.com/nccl/nccl-download "NCCL")

```
sudo dpkg -i nccl-repo-<version>.deb
sudo apt install libnccl2 libnccl-dev
```

## 2. Install Bazel 0.17.2

See [installing Bazel](https://docs.bazel.build/versions/master/install-ubuntu.html "installing Bazel")

## 3. Build TensorFlow-GPU

### 3.1 Preparation

3.1.1. Install python3-distutils 
```
sudo apt-get install python3-distutils
```
pip 
```
sudo apt install python3-dev python3-pip
```

```
pip install -U pip six numpy wheel mock
pip install -U keras_applications==1.0.5 — no-deps
pip install -U keras_preprocessing==1.0.3 — no-deps
```

### 3.2 Download and build

See [Tensorflow-gpu](https://www.tensorflow.org/install/source "tensorflow-gpu")

```git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
git checkout branch_name  # r1.9, r1.10, etc.
```

3.2.2 Test it with bazel.

```
bazel test -c opt — //tensorflow/… -//tensorflow/compiler/… -//tensorflow/contrib/lite/…
```

3.2.3 Configure TensorFlow build by running

```
./configure.
```

> Specify the following options:
```
Please specify the location of python. [Default is /usr/bin/python]: /usr/bin/python3.6

`Do you wish to build TensorFlow with CUDA support? [y/N]: Y`

`Please specify the CUDA SDK version you want to use, e.g. 7.0. [Leave empty to default to CUDA 10.0]: 10.0`

`Please specify the location where CUDA 10 toolkit is installed. Refer to README.md for more details. [Default is /usr/local/cuda]: /usr/local/cuda-10.0`

`Please specify the cuDNN version you want to use. [Leave empty to default to cuDNN 7.0]: 7.4.2`

`Please specify the location where cuDNN 7 library is installed. Refer to README.md for more details. [Default is /usr/local/cuda-9.2]: /usr/lib/x86_64-linux-gnu`

`Please specify the NCCL version you want to use. [Leave empty to default to NCCL 1.3]: 2.4.2`

`Please specify the location where NCCL 2 library is installed. Refer to README.md for more details. [Default is /usr/local/cuda-10.0]: /usr/local/cuda-10.0/targets/x86_64-linux`
```

So... On my laptop, the configuration took about 7 hours.

3.2.4 Build and install TensorFlow.

Run
```
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
```
Be ready to entertain yourself for an hour while building process is in progress. As a result pip package will be placed at /tmp/tensorflow_pkg.
We are almost done — run from your venv and enjoy

```
pip install /tmp/tensorflow_pkg/tensorflow-version-<your_version>.whl
```

## 4. Download Object Detection repository

If you don’t have git installed, install git with following command:

```
sudo apt install -y git
```

Now we will clone Tensorflow models repositories:

```
git clone https://github.com/tensorflow/models
```

## 5. Install Object Detection API

Now we will install the APU using this guide:https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md

So let’s install necessary libraries:
```
sudo apt-get install protobuf-compiler python-pil python-lxml python-tk
pip3 install --user Cython
pip3 install --user contextlib2
pip3 install --user jupyter
pip3 install --user matplotlib
```

## 6. Install COCO API

```
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
cp -r pycocotools /models/research/
```

## 7. Install Protoc

See https://gist.github.com/sofyanhadia/37787e5ed098c97919b8c593f0ec44d8

After downloading run following command:

```
# From /models/research/
protoc object_detection/protos/*.proto --python_out=.
```

## 8. Build and Install research

```
python3 setup.py build
python3 setup.py install
```

## 9. Install openCV

```
https://pypi.org/project/opencv-python/
```

## 10. Install numPy

```
python3 -m pip install numpy

```

## 11. Change path object detection in source code

```
PATH_OBJECT_DETECTION = "<full path>/models/research/object_detection/"
```

## 12. Download model

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

## 13. Start

```
python3 main.py
```

## Refernses:

1. [Click 1](https://medium.com/@vitali.usau/install-cuda-10-0-cudnn-7-3-and-build-tensorflow-gpu-from-source-on-ubuntu-18-04-3daf720b83fe)

2. [Click 2](https://gist.github.com/ljaraque/d18d3dd198dcff3bc40cbe91889564d0)