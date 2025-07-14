# ASR Model

This project implements an Automatic Speech Recognition (ASR) system with PyTorch Lightning.
This is an ASR TOOLKIT designed for greater simplicity and clarity.
Transformer Decoder and CTC Decoding need more updates.

## Project Structure

```
Jcore
├── lm  # DIRECTORY WHERE LM IS SAVED
│   
├── models
│   ├── $ project_name
|   │   ├── data
|   |   │   ├── dataset.py
|   |   │   └── datamodule.py
|   │   ├── models  # CHECKPOINTS ARE SAVED IN HERE
|   │   ├── modules
|   |   │   ├── conformer  # MODULE CODES RELATED TO CONFORMER
|   |   │   ├── decoder  # MODULE CODES RELATED TO DECODER
|   |   │   └── transformer  # MODULE CODES RELATED TO TRANSFORMER
│   └── stats  # FILES RELATED TO FEATURE NORMALIZATION
├── requirements.txt
├── scp  # DATA IS ACCESSED WITH .scp FORMAT
│   ├── $ scp_folder
│   ├── csvtoscp.py
│   ├── extract.py
│   └── update_scp.py
└── util  # ASR UTILITIES FOR ALL PROJECTS

```

## Installation

To install the required dependencies, run:

```
pip install -r requirements.txt
```

## Usage

1. **Training the Model**: To train the model, run the following command:

   ```
   python models/$project_name/train.sh
   ```

2. **Making Predictions**: To make predictions on new audio samples, execute:

   ```
   python models/$project_name/test.sh
   ```


## warp-rnnt installation
For warp-rnnt installation, follow [warp-rnnt-jaeeunbaik](https://github.com/jaeeunbaik/warp-transducer.git) (version for newer GPUs)


### Compilation
warp-transducer has been tested on Ubuntu 16.04 and CentOS 7. Windows is not supported at this time.

First, get the code:
```bash
git clone https://github.com/jaeeunbaik/warp-transducer.git
cd warp-transducer
```
Create a build directory:
```bash
mkdir build
cd build
```
If you have a non-standard CUDA install, add `-DCUDA_TOOLKIT_ROOT_DIR=/path/to/cuda` option to `cmake` so that CMake detects CUDA.

Run cmake and build:
```bash
cmake -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME ..
make
```
> If there is an issue with `CUDA` or `CUDNN` compatibility, please _**deactivate all conda environments**_ and try building again.

### Test
**Important:** Please run `./build/test_gpu` before using on GPU.
To run the tests, make sure the CUDA libraries are in your `LD_LIBRARY_PATH` (or `DYLD_LIBRARY_PATH` for OSX).