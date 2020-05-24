# AASMA
## Deep Q Learning Drone Patroller

## Installation (Makefile)

To install aasma, run 'make'.
To uninstall it, run 'make uninstall'.
You can uninstall and re-install with 'make reset'.

For example, when running the project for the first time do:

```console
make reset
```

All models were trained on a NVIDIA GTX 1070.

## Environment

The AASMA Environment is imported directly by whatever scripts need it.
The environment is a grid world containing 5 types of cells:

* Mountains
* Fires
* Green cells
* Yellow cells
* Red cells

The goal is to minimize the occurrence of fires which are
the evolution of a red cell.

The evolution of the cells is **Green -> Yellow -> Red -> Fires -> Green**.
Mountains do not change.

The configuration of the environment can be altered in the file **config.json**.

The environment can be seen below.

<p align="center">
  <img width="200" src="https://github.com/gm0l74/AASMA/blob/master/images/env.png">
</p>

## Single Agent

The agent related files are responsible for controlling the brain of the drone.

The Agent's type can be one of **random** or **drl** or **reactive**.

The *Random Agent* just chooses a random action at any given time.
The *Reactive Agent* sees the environment, all sees the closest red cells.
Only then it tries to eliminate them.

The *DRL Agent*, however, is controlled by a neural net which receives
an image of the entire environment as input.
It then has to decide which action is the best.

To train the neural net run:
```console
python aasma/train.py
```

You can visualize the input passed to the neural net by running:
```console
python aasma/grabber.py
```

<p align="center">
  <img width="200" src="https://github.com/gm0l74/AASMA/blob/master/images/s_agent.gif">
</p>

To run the project with an already trained model do:
```console
python run.py [drl|random|reactive] [single|multi] <path>
```

...in which <path> is only required if you are using DRL.
For these situations <path> = agents/saved_models

## Multi Agent

Multi agent training is made on top of a model trained in single agent.
Hence the use of *transfer learning* in this project.
To train a multi agent system do:
```console
python train_multi.py agents/saved_models
```

The multi agent system only supports two characters as is.
Future work may be done to extend the support to more simultaneous agents.

<p align="center">
  <img width="300" src="https://github.com/gm0l74/AASMA/blob/master/images/m_agent.gif">
</p>

## Comparisons
To compare all models, one should execute:
```console
python h2h.py
```
