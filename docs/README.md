# CODE : Confident Ordinary Differential Editing
<br>

[**Project Website**](https://vita-epfl.github.io/CODE/) | [![](https://img.shields.io/badge/paper-arXiv-red)](https://arxiv.org/abs/2408.12418v1) | <a target="_blank" href="https://colab.research.google.com/github/vita-epfl/CODE/blob/main/CODE.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

Official PyTorch implementation of **CODE: Confident Ordinary Differential Editing** (2024).

<p align="center">
<img src="https://github.com/vita-epfl/CODE/blob/main/docs/images/main_figure.png" style="width: 50%"/>
<img src="https://github.com/vita-epfl/CODE/blob/main/docs/images/CODE.gif" style="width: 75%"/>
</p>

## Overview
CODE aims to handle guidance image that are Out-of-Distribution in a systematic manner. The key idea is to reverse stochastic process of SDE-based generative models, using the associated Probability Flow ODE in combination with a Confidence Based Clipping, and to make score-based updates in the latent spaces as we use the ODE to generate new images, method as illustrated in the figure below. Given an input image for editing, such as a stroke painting or a corrupted low-quality image, we can make the artifacts undetectable, while preserving the semantics of the image. CODE offers a natural and grounded method to balance the trade-off realism-fidelity of the generated outputs. The user can arbitrarily choose to increase realism in the image or to conserve more of the image guidance. 


<p align="center">
<img src="https://github.com/vita-epfl/CODE/blob/main/docs/images/Code_2.png" />
</p>

## Getting Started

### Creating the environment

Please run,
```
conda env create -f code/environment.yaml
```
Then activate the environment,
```
conda activate code
```

### Generating Images

To generate images, please update the celebahq_hugginface.yaml config file according to your needs, then run,
```
python -m code.main.py --trainer=celebahq_hugginface
```

### Metrics

To compute metrics, first indicates the folder with the generated images on code/metrics/filter_data.py.
Then run,
```
python code/metrics/filter_data.py
```

Then run,
```
bash code/metrics/calculate_all_metrics.sh
```

## References

If you find this repository useful for your research, please cite the following work.

```
@misc{vandelft2024codeconfidentordinarydifferential,
      title={CODE: Confident Ordinary Differential Editing}, 
      author={Bastien van Delft and Tommaso Martorella and Alexandre Alahi},
      year={2024},
      eprint={2408.12418},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.12418}, 
}
```

### SDEdit

For all SDEdit experiment we used the official implementation available at [https://github.com/ermongroup/SDEdit]()
