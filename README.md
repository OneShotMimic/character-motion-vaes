## Quick Start

This library should run on Linux, Mac (Intel version), or Windows. It is used for motion data structure pre-testing in Single-shot Motion Mimicking for Physics-based Character Animation project. 

## Install Requirements

```bash
# TODO: Create and activate virtual env

cd MotionVAEs
pip install -r requirements
NOTE: installing pybullet requires Visual C++ 14 or higher. You can get it from here: https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

## Training data requirement

All the data should be stored in ```{data_dir}```. 

For a source labeled as ```{label}```, its motion data should be stored in ```{data_dir}/x_{label}.pt``` and its control parameters should in ```{data_dir}/actor_{label}.pt```. Predicted control parameters will be stored in ```{data_dir}/pred_{label}.pt```.

The motion data is a sequence of 1\*267 vectors, and the control and output parameters are sequences of 1\*32 vectors. 

## Train a motion parser
```
# Specify the data directory, train label and testing label at the beginning of motion_parser/train_parser.py. 
# Learning rate and number of epochs are specified in main function.

python motion_parser/train_parser.py
```
## Modify model structure.

The model of motion parser is stored in **motion_parser/models.py** 
