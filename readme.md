[![Build Status](https://travis-ci.com/cenkcorapci/fashion_mnist.svg?branch=master)](https://travis-ci.com/cenkcorapci/fashion_mnist)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![GitHub last commit](https://img.shields.io/github/last-commit/cenkcorapci/fashion_mnist)
[![Requirements Status](https://requires.io/github/cenkcorapci/fashion_mnist/requirements.svg?branch=master)](https://requires.io/github/cenkcorapci/fashion_mnist/requirements/?branch=master)

# Image Classification on fashion mnist
## Data Set
Classification models that have been tried on [Zalando's Fashion MNIST data set.](https://github.com/zalandoresearch/fashion-mnist)
- Data set contains images that belongs to 10 different categories of clothing.
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

### Confusion Matrix for the best model

|Metric   |top               |trouser           |pullover          |dress             |coat              |sandal            |shirt             |sneaker           |bag               |ankle boot        |accuracy|macro avg         |weighted avg      |
|---------|------------------|------------------|------------------|------------------|------------------|------------------|------------------|------------------|------------------|------------------|--------|------------------|------------------|
|precision|0.8746465598491989|0.9979838709677419|0.9180651530108588|0.9482071713147411|0.9221789883268483|0.9870259481037924|0.8840579710144928|0.9719157472417251|0.9950099800399201|0.9681274900398407|0.9468  |0.9467218879909162|0.9467218879909161|
|recall   |0.928             |0.99              |0.93              |0.952             |0.948             |0.989             |0.793             |0.969             |0.997             |0.972             |0.9468  |0.9468            |0.9468            |
|f1-score |0.9005337214944202|0.9939759036144579|0.9239940387481372|0.9500998003992016|0.9349112426035503|0.988011988011988 |0.8360569319978914|0.9704556835252879|0.9960039960039959|0.9700598802395209|0.9468  |0.9464103186638452|0.9464103186638451|
|support  |1000.0            |1000.0            |1000.0            |1000.0            |1000.0            |1000.0            |1000.0            |1000.0            |1000.0            |1000.0            |0.9468  |10000.0           |10000.0           |

![confussion](https://raw.githubusercontent.com/cenkcorapci/fashion_mnist/master/images/confussion.png)

- Seems like _tops_ are mistaken with _shirts_ quite often. Maybe another classifier can be trained to differentiate between these two classes and be used with the selected model.
- Confussion matrices looks similar across models, so i didn't try things like [ensembling](https://en.0wikipedia.org/wiki/Ensemble_learning)
- Didn't try different combinations of augmentations, something like grid search can be done for finding a good combination of augmentations in future.

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
The docker image is based on [tensorflow/tensorflow:latest-gpu-py3-jupyter](https://hub.docker.com/r/tensorflow/tensorflow/)
which supports GPUs. Check out [their docs](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/dockerfiles) for more info on how to enable gpu support.
### Jupyter
- `experiment.ipynb` is the starting point.
- `config.py` contains paths for model checkpoints and TensorBoard logs.
    - Path for model checkpoints folder can be set with `DL_MODELS_PATH` environment variable.
    - Path for TensorBoard logs folder can be set with `TB_LOGS_PATH` environment variable.
