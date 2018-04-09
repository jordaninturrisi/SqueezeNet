# SqueezeNet Keras Implementation
Keras implementation of [SqueezeNet](https://arxiv.org/pdf/1602.07360.pdf) (arXiv 1602.07360) using the Keras functional API.

Iandola, F.N., Han, S., Moskewicz, M.W., Ashraf, K., Dally, W.J. and Keutzer, K., 2016. SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and< 0.5 MB model size. arXiv preprint arXiv:1602.07360.

SqueezeNet is a small model of AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size.
The original model was implemented in [caffe](https://github.com/DeepScale/SqueezeNet).


## Reference
[SqueezeNet Keras Implementation](https://github.com/DT42/squeezenet_demo)
Differences:
- Create function `fire_module` to simplify code
- Create variables for data format, kernel initialisation, activation function for easier modifications


## Result
This repository contains only the Keras implementation of the model.

The model is tested on CIFAR-10 & CIFAR-100 and achieves the following performance:
```
loss: 0.xxxx - acc: 0.xxxx - val_loss: 0.xxxx - val_acc: xxxx
```


## Model Visualization
![](SqueezeNet.png)
