# MiniLearn
On-Device Learning for Low-Power IoT Devices

This is a repor for the paper Minilearn as part of International Conference on Embedded Wireless Systems and Network (EWSN) 2022

I'm using the CMSIS-NN libray as submodule, you should clone using git clone --recurse-submodules or if you already cloned use git submodule update --init --recursive

I have create a simplified version that can be compiled on local computer, if you need to cross-compile there board specific steps to follow.

There are three examples for three datasets: CIFAR10, WISDM (HAR), and google keyspotting.

In order to complie you need to check the Makefile (I use relateve folder refence) so you should need to just execute make.

The makefile create a local executable with the name Minilearn, to execute type ./MiniLearn 

In CIFAR10 and HAR example the strucure looks like 

```bash
├── src                     # Source files 
├── obj                     # Obj files from other lib

```

The keyspotting depends on different libraries and more steps is needed to make it work
