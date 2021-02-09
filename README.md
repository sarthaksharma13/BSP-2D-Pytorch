## Table of contents
* [General info](#general-info)
* [Setup](#setup)
* [Usage](#general-info)
* [Results](#results)
* [References](#references)

## General info
This repo deals with the 2D experiments for BSP-Net presented in the work **BSP-Net: Generating Compact Meshes via Binary Space Partitioning** by Chen et.al
The repo is modification from the original work as an option to include batch normalization layers has been added. 
- phase 0 : continuous for better convergence
- phase 1 : hard discretization for BSP.	

## Setup
To set up the environment run,
`conda create --name <env> --file requirements.txt`

## Usage
- First generate the data by running the script ```python create_dataset.py``` in /data folder.
- Run ``` sh train.sh ``` to start training, by specifying different arguments.
- All the experiments would be saved in \exp folder.

## Results

## References
This implementation was referred from the original code released by authors here [Project page](https://bsp-net.github.io)

