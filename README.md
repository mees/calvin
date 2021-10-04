# CALVIN


[<b>CALVIN - A benchmark for Language-Conditioned Policy Learning for Long-Horizon Robot Manipulation Tasks</b>](https://arxiv.org/abs/foo)
[Oier Mees](https://www.oiermees.com/), [Lukas Hermann](http://www2.informatik.uni-freiburg.de/~hermannl/), Erick Rosete, [Wolfram Burgard](http://www2.informatik.uni-freiburg.de/~burgard)

 We present **CALVIN** (**C**omposing **A**ctions from **L**anguage and **Vi**sio**n**), an open-source simulated benchmark to learn long-horizon language-conditioned tasks.
Our aim is to make it possible to develop agents that can solve many robotic manipulation tasks over a long horizon, from onboard sensors, and specified only via human language. CALVIN tasks are more complex in terms of sequence length, action space, and language than existing vision-and-language task datasets and supports flexible specification of sensor
suites.

![](media/teaser.png)

# :computer:  Quick Start
To begin, clone this repository locally
```bash
git clone https://github.com/mees/calvin.git
$ export CALVIN_ROOT=$(pwd)/calvin

```
Install requirements:
```bash
$ cd $CALVIN_ROOT
$ virtualenv -p $(which python3) --system-site-packages calvin_env # or use conda
$ source calvin_env/bin/activate
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

Download dataset:
```bash
$ cd $CALVIN_ROOT/data
$ sh download_data.sh
```

Train baseline models:
```bash
$ cd $CALVIN_ROOT
$ python train.py
```

## Citation

If you find the dataset or code useful, please cite:

```
@article{calvin21,
author = {Oier Mees and Lukas Hermann and Erick Rosete and Wolfram Burgard},
title = {CALVIN - A benchmark for Language-Conditioned Policy Learning for Long-Horizon Robot Manipulation Tasks},
journal={arXiv preprint arXiv:foo},
year = 2020,
}
```

## License

MIT License
