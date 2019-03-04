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
* 6th Generation Intel® Core™ processor with Intel® Iris® Pro graphics or Intel® HD Graphics

### Software
* [Ubuntu\* 16.04 LTS](http://releases.ubuntu.com/16.04/)<br><br>
*Note*: We recommend using a 4.14+ kernel to use this software. Run the following command to determine your kernel version:
```
uname -a
```
* OpenCL™ Runtime Package
* Intel® Distribution of OpenVINO™ toolkit R5 Release

## Setup

### Install Intel® Distribution of OpenVINO™ Toolkit
Refer to [Install the Intel® Distribution of the OpenVINO™ toolkit for Linux*](https://software.intel.com/en-us/articles/OpenVINO-Install-Linux)  for more information about how to install and setup the toolkit.

The software requires the installation of the OpenCL™ Runtime package to run inference on the GPU, as indicated in the following instructions. It is not mandatory for CPU inference.

## How It Works

The application uses a video source, such as a camera, to grab frames, and then uses three different Deep Neural Networks (DNNs) to process the data. The first network detects faces, and if successful, passes the result to the second neural network.

The second neural network is then used to determine if the machine operator is watching the machine (i.e., if the operator's head is facing towards the camera). If the machine operator is watching the machine, **Watching: 1** is displayed on the screen; otherwise, **Watching: 0** is displayed.

Finally, if the proper head position has been detected, the third neural network determines the emotion of the operator's face.


The data can then optionally be sent to a MQTT machine to machine messaging server, as part of an industrial data analytics system.

The DNN models used in this application are Intel® optimized models that are part of the Intel® Distribution of OpenVINO™ toolkit.

Find the optimized models in these locations:

- `/opt/intel/computer_vision_sdk/deployment_tools/intel_models/face-detection-adas-0001`
- `/opt/intel/computer_vision_sdk/deployment_tools/intel_models/head-pose-estimation-adas-0001`
- `/opt/intel/computer_vision_sdk/deployment_tools/intel_models/emotions-recognition-retail-0003`

![Code organization](./images/arch3.png)

The program creates three threads for concurrency:

- Main thread that performs the video I/O
- Worker thread that processes video frames using the deep neural networks
- Worker thread that publishes any MQTT messages

## Install the Dependencies

```
sudo apt-get install mosquitto mosquitto-clients
```

## Set the Build Environment

Configure the environment to use the Intel® Distribution of OpenVINO™ toolkit one time per session by running the following command:
```
    source /opt/intel/computer_vision_sdk/bin/setupvars.sh
```

## Build the Code

Change the current directory to the location of your git-cloned application code. For example:
```
    cd machine-operator-monitor-cpp
```

If no `build` directory exists, create one:
```
    mkdir build
```

Change to  the `build` directory:
```
    cd build
```

Run the following commands:
```
    cmake ..
    make
```

The build commands produce a `monitor` application executable.

## Running the Code

To see a list of the various options:
```
    ./monitor -h
```

To run the application with the needed models using the webcam:
```
    ./monitor -m=/opt/intel/computer_vision_sdk/deployment_tools/intel_models/face-detection-adas-0001/FP32/face-detection-adas-0001.bin -c=/opt/intel/computer_vision_sdk/deployment_tools/intel_models/face-detection-adas-0001/FP32/face-detection-adas-0001.xml -sm=/opt/intel/computer_vision_sdk/deployment_tools/intel_models/emotions-recognition-retail-0003/FP32/emotions-recognition-retail-0003.bin -sc=/opt/intel/computer_vision_sdk/deployment_tools/intel_models/emotions-recognition-retail-0003/FP32/emotions-recognition-retail-0003.xml -pm=/opt/intel/computer_vision_sdk/deployment_tools/intel_models/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.bin -pc=/opt/intel/computer_vision_sdk/deployment_tools/intel_models/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml
```
The user can choose different confidence levels for both face and emotion detection by using `--faceconf or -f,` and `--moodconf or -mc` command line parameters respectively. By default, both of these parameters are set to `0.5` (i.e., at least `50%` detection confidence is required in order for the returned inference result to be considered valid).

### Hardware acceleration

This application can leverage hardware acceleration in the Intel® Distribution of OpenVINO™ toolkit by using the `-b` and `-t` parameters.

For example, to use the Intel® Distribution of OpenVINO™ toolkit backend with the GPU in 32-bit mode:
```
    ./monitor -m=/opt/intel/computer_vision_sdk/deployment_tools/intel_models/face-detection-adas-0001/FP32/face-detection-adas-0001.bin -c=/opt/intel/computer_vision_sdk/deployment_tools/intel_models/face-detection-adas-0001/FP32/face-detection-adas-0001.xml -sm=/opt/intel/computer_vision_sdk/deployment_tools/intel_models/emotions-recognition-retail-0003/FP32/emotions-recognition-retail-0003.bin -sc=/opt/intel/computer_vision_sdk/deployment_tools/intel_models/emotions-recognition-retail-0003/FP32/emotions-recognition-retail-0003.xml -pm=/opt/intel/computer_vision_sdk/deployment_tools/intel_models/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.bin -pc=/opt/intel/computer_vision_sdk/deployment_tools/intel_models/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml -b=2 -t=1
```

To run the code using 16-bit floats, set the `-t` flag to use the GPU in 16-bit mode, as well as use the FP16 version of the Intel® models:
```
    ./monitor -m=/opt/intel/computer_vision_sdk/deployment_tools/intel_models/face-detection-adas-0001/FP16/face-detection-adas-0001.bin -c=/opt/intel/computer_vision_sdk/deployment_tools/intel_models/face-detection-adas-0001/FP16/face-detection-adas-0001.xml -sm=/opt/intel/computer_vision_sdk/deployment_tools/intel_models/emotions-recognition-retail-0003/FP16/emotions-recognition-retail-0003.bin -sc=/opt/intel/computer_vision_sdk/deployment_tools/intel_models/emotions-recognition-retail-0003/FP16/emotions-recognition-retail-0003.xml -pm=/opt/intel/computer_vision_sdk/deployment_tools/intel_models/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.bin -pc=/opt/intel/computer_vision_sdk/deployment_tools/intel_models/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml -b=2 -t=2
```

To run the code using the VPU, set the `-t` flag to `3` and also use the 16-bit FP16 version of the Intel® models:
```
    ./monitor -m=/opt/intel/computer_vision_sdk/deployment_tools/intel_models/face-detection-adas-0001/FP16/face-detection-adas-0001.bin -c=/opt/intel/computer_vision_sdk/deployment_tools/intel_models/face-detection-adas-0001/FP16/face-detection-adas-0001.xml -sm=/opt/intel/computer_vision_sdk/deployment_tools/intel_models/emotions-recognition-retail-0003/FP16/emotions-recognition-retail-0003.bin -sc=/opt/intel/computer_vision_sdk/deployment_tools/intel_models/emotions-recognition-retail-0003/FP16/emotions-recognition-retail-0003.xml -pm=/opt/intel/computer_vision_sdk/deployment_tools/intel_models/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.bin -pc=/opt/intel/computer_vision_sdk/deployment_tools/intel_models/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml -b=2 -t=3
```

## Sample Videos

There are several sample videos that can be used to demonstrate the capabilities of this application. Download them by running these commands from the `machine-operator-monitor` directory:
```
    mkdir resources
    cd resources
    wget https://github.com/intel-iot-devkit/sample-videos/raw/master/head-pose-face-detection-female.mp4
    wget https://github.com/intel-iot-devkit/sample-videos/raw/master/head-pose-face-detection-male.mp4
    cd ..
```

To then execute the code using one of these sample videos, run the following commands from the `machine-operator-monitor-cpp` directory:
```
    cd build
    ./monitor -m=/opt/intel/computer_vision_sdk/deployment_tools/intel_models/face-detection-adas-0001/FP32/face-detection-adas-0001.bin -c=/opt/intel/computer_vision_sdk/deployment_tools/intel_models/face-detection-adas-0001/FP32/face-detection-adas-0001.xml -sm=/opt/intel/computer_vision_sdk/deployment_tools/intel_models/emotions-recognition-retail-0003/FP32/emotions-recognition-retail-0003.bin -sc=/opt/intel/computer_vision_sdk/deployment_tools/intel_models/emotions-recognition-retail-0003/FP32/emotions-recognition-retail-0003.xml -pm=/opt/intel/computer_vision_sdk/deployment_tools/intel_models/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.bin -pc=/opt/intel/computer_vision_sdk/deployment_tools/intel_models/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml -i=../resources/head-pose-face-detection-female.mp4
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
