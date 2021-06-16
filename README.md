# S3D
Squeeze and Excitation 3D Convolutional Neural Networks for Fall Detection System

## Requirements
Python 3.8 or later with all [requirements.txt](https://github.com/baek2sm/S3D/blob/master/requirements.txt) dependencies installed. To install run:
```bash
$ pip install -r requirements.txt
```

## Performance (2-fold cross validation)
  Dataset|Accuracy(%)|Precision(%)|Recall(%)|F1(%)
  -----|--|--|--|--
  Thermal Simulated Fall [1]|95.45%|97.22%|97.22%|97.22%
  TCL Fall Detection [2]|96.81%|97.09%|93.35%|95.18%

## Pre-trained models
  Dataset|fold|Download link
  --|-|-----
  Thermal Simulated Fall|1|https://drive.google.com/file/d/1ADHLGNFusdMyvfslD-YSFjR1smK1q-Ew/view?usp=sharing
  Thermal Simulated Fall|2|https://drive.google.com/file/d/1FJk3pFx_TL-jHTaKgCVVspX5eFpKo-zi/view?usp=sharing
  TCL Fall Detection|1|https://drive.google.com/file/d/1hKS8Cnw-Y4HoFvFTRop2KO0d0z1wlO_S/view?usp=sharing
  TCL Fall Detection|2|https://drive.google.com/file/d/1q1dFC6Efd8gNxYL-0w0TrZRpOVP9vyBT/view?usp=sharing

## References
[1] Vadivelu, S.; Ganesan, S.; Murthy, O.R.; Dhall, A. Thermal imaging based elderly fall detection. In Proceedings of Asian Conference on Computer Vision, 2016, pp. 541-553.
[2] Kim, D.-E.; Jeon, B.; Kwon, D.-S. 3D Convolutional Neural Networks based Fall Detection with Thermal Camera. The Journal of Korea Robotics Society 2018, 13, pp. 45-54.
