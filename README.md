# Imitation learning [![Build Status](https://travis-ci.org/merantix/imitation-learning.svg?branch=master)](https://travis-ci.org/merantix/imitation-learning)


This repository provides a Tensorflow implementation of the paper 
[End-to-end Driving via Conditional Imitation Learning](http://vladlen.info/papers/conditional-imitation.pdf).

You can find a pre-trained network
[here](https://github.com/carla-simulator/imitation-learning/). 
The repository at hand adds Tensorflow training code. 

There are only a few changes to the setup in the paper:
* We train less steps (we do 190k steps, the paper does 450k steps), but this is configurable.
* The branches for the controller follow the order of the training data.
* We take different weight hyperparameters for the outputs (steer, gas, brake, speed), 
  since the hyperparameters suggested in the paper did not work for us.


### Setup

This repository uses docker images. In order to use it, install [docker](https://docs.docker.com/install/). 
To build the image, use:

```bash
docker build --build-arg base_image=tensorflow/tensorflow:1.12.0-gpu -t imit-learn .
```

If you only need a CPU image, leave out `base_image=tensorflow/tensorflow:1.12.0-gpu`.
So far, we only tested the setup with Python2, which `tensorflow:1.12.0` is based on.

To run a container, use:

```bash
cd <root of this repository>
DOCKER_BASH_HISTORY="$(pwd)/data/docker.bash_history"
touch $DOCKER_BASH_HISTORY

docker run -it --rm --name imit_learn \ 
    -v $(pwd)/imitation:/imitation -v $(pwd)/data:/data \
    -v "$DOCKER_BASH_HISTORY:/root/.bash_history" \
    imit-learn bash
```

Download [dataset](https://github.com/carla-simulator/imitation-learning/#user-content-dataset) (24GB).
Unpack!
Put them into `data/imitation_learning/h5_files/AgentHuman`.

If you don't wanna download all the data right away, you can try on a very small subset
that is contained in this repository. To set it up, run:

```bash
cd <root of this repository>
mkdir data/imitation_learning/h5_files/
cp -r imitation/test/mock_data_181018/imitation_learning/h5_files/ data/imitation_learning/h5_files/
```

### Preprocessing

The preprocessing converts the downloaded h5 files into tfrecord files
so we can easier use them for training with Tensorflow.

During preprocessing, the data is shuffled to a certain degree.
More specifically speaking, it is shuffled the h5 files, but it is not shuffling the frames inside an h5 file.  
Shuffling across files is achieved by using a big shuffle buffer during training.

Run preprocessing using:

 ```bash
mkdir -p /data/imitation_learning/preprocessed/
python /imitation/preprocessor.py --preproc_config_paths=config-preprocess-production.yaml
```

This might run for a while and slow consume a lot of CPU power.
To simply check that the preprocessing code can run, set `--preproc_config_paths=config-preprocess-debug.yaml`.


### Train

In order for the training to run, training and validation data need to be in the right place as described above.
To run training with best hyperparameters on the 
entire dataset, run:

```bash
python trainer.py --config_paths=config-train-production.yaml
```

To debug training, use:

```bash
python trainer.py --config_paths=config-train-debug.yaml
```


### Tests

```bash
pytest
```
