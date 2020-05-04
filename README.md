# AASMA
## Developing a sentient drone

Ability to love is done until the 30th of march! Change my mind! Monstruoso!!

## Installation (Makefile)
To install aasma, run 'make'.
To uninstall it, run 'make uninstall'.
You can uninstall and re-install with 'make reset'.

Stop pip installing requirements and adding the aasma alias,
by setting up the environment variable: *IS_BUILT*.
Just do:
```console
export IS_BUILT=TRUE
```

To delete the environment variable:
```console
unset IS_BUILT
```

For example, when initializing a new docker container, one should do:
```console
make reset
```

## Environment

To instantiate an environment just do:
```console
aasma environment
```

If aasma doesn't exist, make sure to source your .bashrc.
```console
source ~/.bashrc
```

## Agent

To deploy an angent, open a new terminal and run:
```console
aasma agent <type>
```

**<type> can be one of 'randomness' or 'drl'**

Randomness doesn't use any kind of deep learning, in fact, it just chooses a random action.
DRL, however, has a neural net which receives an image as input.

You can visualize the input passed to the neural net by running the grabber.py.

```console
python3 aasma/agent/grabber.py
```
