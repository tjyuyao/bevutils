# Bird's-Eye View Utilities
by Huang Yuyao, Wen YongKun

## Features

1. A differentiable Perspective Transformer Layer inspired by Spatial Transformer Networks. See `bevutils/layers/perspective_transformer.py` and `samples/basic/sample.py`.
2. A simulated dataset. See `bevutils/datasets/simulated.py` and `samples/simulated_dataset/demo_dataset.py`.
3. A traning system inspired by [victoresque's ](https://github.com/victoresque/pytorch-template) and [mmdetection](https://github.com/open-mmlab/mmdetection/)

## Installation

`pip install git+https://github.com/huangyuyao/bevutils.git@master`

## Development

```
git clone https://github.com/huangyuyao/bevutils.git
cd bevutils
pip install -e .
```

## Usage

You can make use of functions, datasets and layers of this project by import the functionalities from anywhere.

If you want to use the capability of training a new network, you can do three things:

    - registry your custom network modules(TBD),
    - write a config file,
    - and call `bevtrain -c config.json` or `bevtest -c config.json` tools to reuse the training process. (TBD)

You can refer to `samples/perspective_transformer_network` as a quick tutorial. (TBD)

## License

This project is licensed under the MIT License. See LICENSE for more details