# Evasion Generative Adversarial Network for Low Data Regimes

A myriad of recent literary works has leveraged generative adversarial networks (GANs) to spawn unseen evasion samples. The purpose is to annex the generated data with the original train set for adversarial training to improve the detection performance of machine learning (ML) classifiers. The quality of generating adversarial samples relies on the adequacy of training data samples. However, in low data regimes like medical diagnostic imaging and cybersecurity, the anomaly samples are scarce in number. This paper proposes a novel GAN design called Evasion Generative Adversarial Network (EVAGAN) that is more suitable for low data regime problems that use oversampling for detection improvement of ML classifiers. EVAGAN not only can generate evasion samples, but its discriminator can act as an evasion aware classifier. We have considered Auxiliary Classifier GAN (ACGAN) as a benchmark to evaluate the performance of EVAGAN on cybersecurity (ISCX-2014, CIC-2017 and CIC2018) botnet and computer vision (MNIST) datasets. We demonstrate that EVAGAN outperforms ACGAN for unbalanced datasets with respect to detection performance, training stability and time complexity. EVAGAN's generator quickly learns to generate the low sample class and hardens its discriminator simultaneously. In contrast to ML classifiers that require security hardening after being adversarially trained by GAN generated data, EVAGAN renders it needless. The experimental analysis proves that EVAGAN is an efficient evasion hardened model for low data regimes for the selected cybersecurity and computer vision datasets.

![](EVAGAN.svg "EVAGAN Architecture")

## Prerequisites

Following main libraries were used in the development. However, some other libraries being imported in the header.py file may need to be installed.

- Tensorflow
- Keras
- Numpy
- For the rest of the packages please refer to `header.py` file.

## Manuscript

Full text manuscript can be found [here](https://arxiv.org/abs/2109.08026).

## Dataset

The preprocessed datasets can be downloaded from [here](https://drive.google.com/drive/folders/1sOUEK0Cgpm_P_uxpBydTzGXbJdPISGnU?usp=sharing).

## Cite this Work

```
Randhawa, R. H., Aslam, N., Alauthman, M., & Rafiq, H. (2021). EVAGAN: Evasion Generative Adversarial Network for Low Data Regimes. arXiv preprint arXiv:2109.08026.
```

```
@misc{randhawa2022evagan,
      title={EVAGAN: Evasion Generative Adversarial Network for Low Data Regimes},
      author={Rizwan Hamid Randhawa and Nauman Aslam and Mohammad Alauthman and Husnain Rafiq},
      year={2022},
      eprint={2109.08026},
      archivePrefix={arXiv},
      primaryClass={cs.CR}
}

```
