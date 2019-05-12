# Machine Operator Monitor

| Details            |              |
|-----------------------|---------------|
| Target OS:            |  Ubuntu\* 16.04 LTS   |
| Programming Language: |  C++ |
| Time to Complete:    |  45 min     |

![app image](./images/machine-operator-monitor.png)

## Introduction

This machine operator monitor application is one of a series of reference implementations for Computer Vision (CV) using the Intel® Distribution of OpenVINO™ toolkit. This application is designed for a machine mounted camera system that monitors if the operator is looking at the machine and if his emotional state is detected as angry. It sends an alert if the operator is not watching the machine while it is in operation or if the emotional state of the operator is angry.  It also sends an alert if this combined operator state lasts for longer than a user-defined period of time.

This reference implementation illustrates an example of machine operator safety monitoring.

## Requirements

### Hardware
* 6th to 8th Generation Intel® Core™ processors with Intel® Iris® Pro graphics or Intel® HD Graphics

### Software
* [Ubuntu\* 16.04 LTS](http://releases.ubuntu.com/16.04/)<br><br>
**Note**: We recommend using a 4.14+ kernel to use this software. Run the following command to determine your kernel version:
    ```
    uname -a
    ```
* OpenCL™ Runtime Package
* Intel® Distribution of OpenVINO™ toolkit 2019 R1 Release

## Setup

### Install Intel® Distribution of OpenVINO™ Toolkit
Refer to [Install the Intel® Distribution of the OpenVINO™ toolkit for Linux*](https://software.intel.com/en-us/articles/OpenVINO-Install-Linux)  for more information about how to install and setup the toolkit.

The software requires the installation of the OpenCL™ Runtime package to run inference on the GPU, as indicated in the following instructions. It is not mandatory for CPU inference.

## How It Works

The application uses a video source, such as a camera, to grab frames, and then uses three different Deep Neural Networks (DNNs) to process the data. The first network detects faces, and if successful, passes the result to the second neural network.

The second neural network is then used to determine if the machine operator is watching the machine (i.e., if the operator's head is facing towards the camera). If the machine operator is watching the machine, **Watching: 1** is displayed on the screen; otherwise, **Watching: 0** is displayed.

Finally, if the proper head position has been detected, the third neural network determines the emotion of the operator's face.


The data can then optionally be sent to a MQTT machine to machine messaging server, as part of an industrial data analytics system.

The DNN models can be downloaded using `downloader.py` which is the part of the Intel® Distribution of OpenVINO™ toolkit.


![Code organization](./images/arch3.png)

The program creates three threads for concurrency:

- Main thread that performs the video I/O
- Worker thread that processes video frames using the deep neural networks
- Worker thread that publishes any MQTT messages

## Install the Dependencies

```
sudo apt-get install mosquitto mosquitto-clients
```

## Download the model

This application uses the **face-detection-adas-0001**, **emotions-recognition-retail-0003** and **head-pose-estimation-adas-0001** Intel® model, that can be downloaded using the model downloader. The model downloader downloads the .xml and .bin files that will be used by the application.

Steps to download .xml and .bin files:
- Go to the **model_downloader** directory using the following command:
    ```
    cd /opt/intel/openvino/deployment_tools/tools/model_downloader
    ```

- Specify which model to download with __--name__:
    ```
    sudo ./downloader.py --name face-detection-adas-0001
    sudo ./downloader.py --name head-pose-estimation-adas-0001
    sudo ./downloader.py --name emotions-recognition-retail-0003
    ```
- To download the model for FP16, run the following commands:
    ```
    sudo ./downloader.py --name face-detection-adas-0001-fp16
    sudo ./downloader.py --name head-pose-estimation-adas-0001-fp16
    sudo ./downloader.py --name emotions-recognition-retail-0003-fp16
    ```

 The files will be downloaded inside the following directories respectively:  
 ```
/opt/intel/openvino/deployment_tools/tools/model_downloader/Transportation/object_detection/face/pruned_mobilenet_reduced_ssd_shared_weights/dldt/
/opt/intel/openvino/deployment_tools/tools/model_downloader/Retail/object_attributes/emotions_recognition/0003/dldt/ 
/opt/intel/openvino/deployment_tools/tools/model_downloader/Transportation/object_attributes/headpose/vanilla_cnn/dldt/ 
 ```



## Set the Build Environment

Configure the environment to use the Intel® Distribution of OpenVINO™ toolkit one time per session by running the following command:
```
source /opt/intel/openvino/bin/setupvars.sh
```

## Build the Code

Go to `machine-operator-monitor-cpp` directory and build the application:
```
mkdir -p build && cd build
cmake ..
make
```

## Running the Code

To see a list of the various options:
```
./monitor -h
```

To run the application with the needed models using the webcam:
```
./monitor -m=/opt/intel/openvino/deployment_tools/tools/model_downloader/Transportation/object_detection/face/pruned_mobilenet_reduced_ssd_shared_weights/dldt/face-detection-adas-0001.bin -c=/opt/intel/openvino/deployment_tools/tools/model_downloader/Transportation/object_detection/face/pruned_mobilenet_reduced_ssd_shared_weights/dldt/face-detection-adas-0001.xml -sm=/opt/intel/openvino/deployment_tools/tools/model_downloader/Retail/object_attributes/emotions_recognition/0003/dldt/emotions-recognition-retail-0003.bin -sc=/opt/intel/openvino/deployment_tools/tools/model_downloader/Retail/object_attributes/emotions_recognition/0003/dldt/emotions-recognition-retail-0003.xml -pm=/opt/intel/openvino/deployment_tools/tools/model_downloader/Transportation/object_attributes/headpose/vanilla_cnn/dldt/head-pose-estimation-adas-0001.bin -pc=/opt/intel/openvino/deployment_tools/tools/model_downloader/Transportation/object_attributes/headpose/vanilla_cnn/dldt/head-pose-estimation-adas-0001.xml
```
The user can choose different confidence levels for both face and emotion detection by using `--faceconf or -f,` and `--moodconf or -mc` command line parameters respectively. By default, both of these parameters are set to `0.5` (i.e., at least `50%` detection confidence is required in order for the returned inference result to be considered valid).

### Hardware acceleration

This application can leverage hardware acceleration in the Intel® Distribution of OpenVINO™ toolkit by using the `-b` and `-t` parameters.

For example, to use the Intel® Distribution of OpenVINO™ toolkit backend with the GPU in 32-bit mode:
```
./monitor -m=/opt/intel/openvino/deployment_tools/tools/model_downloader/Transportation/object_detection/face/pruned_mobilenet_reduced_ssd_shared_weights/dldt/face-detection-adas-0001.bin -c=/opt/intel/openvino/deployment_tools/tools/model_downloader/Transportation/object_detection/face/pruned_mobilenet_reduced_ssd_shared_weights/dldt/face-detection-adas-0001.xml -sm=/opt/intel/openvino/deployment_tools/tools/model_downloader/Retail/object_attributes/emotions_recognition/0003/dldt/emotions-recognition-retail-0003.bin -sc=/opt/intel/openvino/deployment_tools/tools/model_downloader/Retail/object_attributes/emotions_recognition/0003/dldt/emotions-recognition-retail-0003.xml -pm=/opt/intel/openvino/deployment_tools/tools/model_downloader/Transportation/object_attributes/headpose/vanilla_cnn/dldt/head-pose-estimation-adas-0001.bin -pc=/opt/intel/openvino/deployment_tools/tools/model_downloader/Transportation/object_attributes/headpose/vanilla_cnn/dldt/head-pose-estimation-adas-0001.xml -b=2 -t=1
```

To run the code using 16-bit floats, set the `-t` flag to use the GPU in 16-bit mode, as well as use the FP16 version of the Intel® models:
```
./monitor -m=/opt/intel/openvino/deployment_tools/tools/model_downloader/Transportation/object_detection/face/pruned_mobilenet_reduced_ssd_shared_weights/dldt/face-detection-adas-0001-fp16.bin -c=/opt/intel/openvino/deployment_tools/tools/model_downloader/Transportation/object_detection/face/pruned_mobilenet_reduced_ssd_shared_weights/dldt/face-detection-adas-0001-fp16.xml -sm=/opt/intel/openvino/deployment_tools/tools/model_downloader/Retail/object_attributes/emotions_recognition/0003/dldt/emotions-recognition-retail-0003-fp16.bin -sc=/opt/intel/openvino/deployment_tools/tools/model_downloader/Retail/object_attributes/emotions_recognition/0003/dldt/emotions-recognition-retail-0003-fp16.xml -pm=/opt/intel/openvino/deployment_tools/tools/model_downloader/Transportation/object_attributes/headpose/vanilla_cnn/dldt/head-pose-estimation-adas-0001-fp16.bin -pc=/opt/intel/openvino/deployment_tools/tools/model_downloader/Transportation/object_attributes/headpose/vanilla_cnn/dldt/head-pose-estimation-adas-0001-fp16.xml -b=2 -t=2
```

To run the code using the VPU, set the `-t` flag to `3` and use the 16-bit FP16 version of the Intel® models:
```
./monitor -m=/opt/intel/openvino/deployment_tools/tools/model_downloader/Transportation/object_detection/face/pruned_mobilenet_reduced_ssd_shared_weights/dldt/face-detection-adas-0001-fp16.bin -c=/opt/intel/openvino/deployment_tools/tools/model_downloader/Transportation/object_detection/face/pruned_mobilenet_reduced_ssd_shared_weights/dldt/face-detection-adas-0001-fp16.xml -sm=/opt/intel/openvino/deployment_tools/tools/model_downloader/Retail/object_attributes/emotions_recognition/0003/dldt/emotions-recognition-retail-0003-fp16.bin -sc=/opt/intel/openvino/deployment_tools/tools/model_downloader/Retail/object_attributes/emotions_recognition/0003/dldt/emotions-recognition-retail-0003-fp16.xml -pm=/opt/intel/openvino/deployment_tools/tools/model_downloader/Transportation/object_attributes/headpose/vanilla_cnn/dldt/head-pose-estimation-adas-0001-fp16.bin -pc=/opt/intel/openvino/deployment_tools/tools/model_downloader/Transportation/object_attributes/headpose/vanilla_cnn/dldt/head-pose-estimation-adas-0001-fp16.xml -b=2 -t=3
```
#### Run the application on FPGA:

Before running the application on the FPGA, program the AOCX (bitstream) file.
Use the setup_env.sh script from [fpga_support_files.tgz](http://registrationcenter-download.intel.com/akdlm/irc_nas/12954/fpga_support_files.tgz) to set the environment variables.<br>

```
source /home/<user>/Downloads/fpga_support_files/setup_env.sh
```

The bitstreams for HDDL-F can be found under the `/opt/intel/openvino/bitstreams/a10_vision_design_bitstreams` folder. To program the bitstream use the below command:
```
aocl program acl0 /opt/intel/openvino/bitstreams/a10_vision_design_bitstreams/2019R1_PL1_FP11_MobileNet_Clamp.aocx
```

For more information on programming the bitstreams, please refer to https://software.intel.com/en-us/articles/OpenVINO-Install-Linux-FPGA#inpage-nav-11

To run the code using the FPGA, you have to set the `-t` flag to `5`:
```
./monitor -m=/opt/intel/openvino/deployment_tools/tools/model_downloader/Transportation/object_detection/face/pruned_mobilenet_reduced_ssd_shared_weights/dldt/face-detection-adas-0001-fp16.bin -c=/opt/intel/openvino/deployment_tools/tools/model_downloader/Transportation/object_detection/face/pruned_mobilenet_reduced_ssd_shared_weights/dldt/face-detection-adas-0001-fp16.xml -sm=/opt/intel/openvino/deployment_tools/tools/model_downloader/Retail/object_attributes/emotions_recognition/0003/dldt/emotions-recognition-retail-0003-fp16.bin -sc=/opt/intel/openvino/deployment_tools/tools/model_downloader/Retail/object_attributes/emotions_recognition/0003/dldt/emotions-recognition-retail-0003-fp16.xml -pm=/opt/intel/openvino/deployment_tools/tools/model_downloader/Transportation/object_attributes/headpose/vanilla_cnn/dldt/head-pose-estimation-adas-0001-fp16.bin -pc=/opt/intel/openvino/deployment_tools/tools/model_downloader/Transportation/object_attributes/headpose/vanilla_cnn/dldt/head-pose-estimation-adas-0001-fp16.xml -b=2 -t=5
```

## Sample Videos

There are several sample videos that can be used to demonstrate the capabilities of this application. Download them by running these commands from the `machine-operator-monitor-cpp` directory:
```
mkdir resources
cd resources
wget https://github.com/intel-iot-devkit/sample-videos/raw/master/head-pose-face-detection-female.mp4
wget https://github.com/intel-iot-devkit/sample-videos/raw/master/head-pose-face-detection-male.mp4
cd ..
```

To execute the code using one of these sample videos, run the following commands from the `machine-operator-monitor-cpp` directory:
```
cd build
./monitor -m=/opt/intel/openvino/deployment_tools/tools/model_downloader/Transportation/object_detection/face/pruned_mobilenet_reduced_ssd_shared_weights/dldt/face-detection-adas-0001.bin -c=/opt/intel/openvino/deployment_tools/tools/model_downloader/Transportation/object_detection/face/pruned_mobilenet_reduced_ssd_shared_weights/dldt/face-detection-adas-0001.xml -sm=/opt/intel/openvino/deployment_tools/tools/model_downloader/Retail/object_attributes/emotions_recognition/0003/dldt/emotions-recognition-retail-0003.bin -sc=/opt/intel/openvino/deployment_tools/tools/model_downloader/Retail/object_attributes/emotions_recognition/0003/dldt/emotions-recognition-retail-0003.xml -pm=/opt/intel/openvino/deployment_tools/tools/model_downloader/Transportation/object_attributes/headpose/vanilla_cnn/dldt/head-pose-estimation-adas-0001.bin -pc=/opt/intel/openvino/deployment_tools/tools/model_downloader/Transportation/object_attributes/headpose/vanilla_cnn/dldt/head-pose-estimation-adas-0001.xml -i=../resources/head-pose-face-detection-female.mp4
```

### Machine to Machine Messaging with MQTT

To use a MQTT server to publish data, set the following environment variables before running the program:
```
export MQTT_SERVER=localhost:1883
export MQTT_CLIENT_ID=cvservice
```

Change the `MQTT_SERVER` to a value that matches the MQTT server to which you are connected.

Change the `MQTT_CLIENT_ID` to a unique value for each monitoring station to track the data for individual locations. For example:
```
export MQTT_CLIENT_ID=machine1337
```

To monitor the MQTT messages sent to your local server, ensure that the `mosquitto`  client utilities are installed. Run the following command in a new terminal while the application is running:
```
mosquitto_sub -t 'machine/safety'
```
