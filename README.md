# S3D
Squeeze and Excitation 3D Convolutional Neural Networks for Fall Detection System. This project is currently being written in a paper.

<div align="center">
  <img width="486" alt="se_3d" src="https://user-images.githubusercontent.com/30026090/141882787-5ec2f40c-d857-4347-83f8-4fa098accf34.png">
  <br>
  <span>3D Squeeze and Excitasation block</span>
  <img width="486" alt="s3d" src="https://user-images.githubusercontent.com/30026090/141882885-f2eac6fd-0e5c-4824-9b04-c61e9ecc646d.png">
  <span>Proposed S3D model</span>
</div>

## Requirements
Python 3.8 or later with all [requirements.txt](https://github.com/baek2sm/S3D/blob/master/requirements.txt) dependencies installed. To install run:
```bash
$ pip install -r requirements.txt
```

## Performance (2-fold cross validation)
  Dataset|Accuracy(%)|Precision(%)|Recall(%)|F1(%)
  -----|--|--|--|--
  Thermal Simulated Fall [1]|95.45%|97.14%|97.14%|97.14%
  TCL Fall Detection [2]|96.89%|96.59%|94.06%|95.30%
  eHomeSeniors [3]|98.91%|98.46%|99.33%|98.89%

## Pre-trained models
  Dataset|fold|Download link
  --|-|-----
  Thermal Simulated Fall|1|https://drive.google.com/file/d/1gc6cbFru5ALWUkt66cFNieDUFhVA69K4/view?usp=sharing
  Thermal Simulated Fall|2|https://drive.google.com/file/d/1FdpPsc8J10qalajITnyDndD9XN2L6Q_z/view?usp=sharing
  TCL Fall Detection|1|https://drive.google.com/file/d/16Kq9ppEZJzIrVtkZR-6ep1Td67HAaqXb/view?usp=sharing
  TCL Fall Detection|2|https://drive.google.com/file/d/1pM9FgUlYyAlakCYA_jnOrrjm5t0jtPfy/view?usp=sharing
  eHomeSeniors|1|https://drive.google.com/file/d/1Qp-S6L3ycsJrnPySJ1I0FfYEJcxwCuAs/view?usp=sharing
  eHomeSeniors|2|https://drive.google.com/file/d/1kQ2Zho6-c2fQnJ7hIU656r5BpRlsCBPH/view?usp=sharing

## References
- [1] Vadivelu, S.; Ganesan, S.; Murthy, O.R.; Dhall, A. Thermal imaging based elderly fall detection. In Proceedings of Asian Conference on Computer Vision, 2016, pp. 541-553.
- [2] Kim, D.-E.; Jeon, B.; Kwon, D.-S. 3D Convolutional Neural Networks based Fall Detection with Thermal Camera. The Journal of Korea Robotics Society 2018, 13, pp. 45-54.
- [3] Riquelme, F.; Espinoza, C.; Rodenas, T.; Minonzio, J.-G.; Taramasco, C. eHomeSeniors dataset: an infrared thermal sensor dataset for automatic fall detection research. Sensors 2019, 19, p. 4565.
