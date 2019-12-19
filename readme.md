[![Build Status](https://travis-ci.com/cenkcorapci/fashion_mnist.svg?branch=master)](https://travis-ci.com/cenkcorapci/fashion_mnist)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![GitHub last commit](https://img.shields.io/github/last-commit/cenkcorapci/fashion_mnist)
[![Requirements Status](https://requires.io/github/cenkcorapci/fashion_mnist/requirements.svg?branch=master)](https://requires.io/github/cenkcorapci/fashion_mnist/requirements/?branch=master)

# Image Classification on fashion mnist
## Data Set
Classification models that have been tried on [Zalando's Fashion MNIST data set.](https://github.com/zalandoresearch/fashion-mnist)
- Data set contains images that belongs to 10 differenct categories of clothing.
- All images are grayscale with dimensions of 28x28
- Data set contains an 60k samples that are evenly distributed between classes.
## Results
- [Stochastic Weight Averaging](https://arxiv.org/abs/1803.05407) has been used in all experiments.
- _rotation_range_, _zoom_range_, _width_shift_range_ ,_height_shift_range_ and _horizontal_flip_
augmentations has been used, you can check the details and tune your selection of augmentations in `experiment.ipynb`

#### Best accuracy scores obtained in validation
- You can see the detailed accuracy results in [tensorboard.dev](https://tensorboard.dev/experiment/k5J2QKE8QI6w81wy1Uyh0w/#scalars)

![accuracy](https://raw.githubusercontent.com/cenkcorapci/fashion_mnist/master/images/accuracies.png)

|model name|optimizer|accuracy|
|---|---|---|
|[CapsuleNet](https://users.aalto.fi/~alexilin/advanced_deep_learning/Capsules.pdf)|Adam|0.9465|
|[Wide ResNet](https://arxiv.org/abs/1605.07146)|Adam|0.9392|
|[MobileNetV2](https://arxiv.org/abs/1801.04381)|Adam|0.9247|
|Simple CNN|Adam|0.9207|
|[ShuffleNet V2](https://arxiv.org/abs/1807.11164)|RMSProp|0.905|
|[ShuffleNet V2](https://arxiv.org/abs/1807.11164) With Cyclic Learning Rate|Adam|0.8964|
|[ShuffleNet V2](https://arxiv.org/abs/1807.11164) With Cyclic Learning Rate|RMSProp|0.906|
|Simple CNN|RMSProp|0.8912|
|[ShuffleNet V2](https://arxiv.org/abs/1807.11164)|Adam|0.8548|


## Usage
### Docker
Integrated travis plugin uploads a docker [image](https://hub.docker.com/r/cenkcorapci/fmnist_models) to docker hub when a new commit gets pushed
to `master` branch. So you can just
```bash
 docker pull cenkcorapci/fmnist_models
```
and then
```bash
docker run --gpus all -it --rm -p 8888:8888 cenkcorapci/fmnist_models
```
### Jupyter
- `experiment.ipynb` is the starting point.
- `config.py` contains paths for model checkpoints and TensorBoard logs.
    - Path for model checkpoints folder can be set with `DL_MODELS_PATH` environment variable.
    - Path for TensorBoard logs folder can be set with `TB_LOGS_PATH` environment variable.
