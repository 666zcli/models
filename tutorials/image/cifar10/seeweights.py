#see the results of quantization
import tensorflow as tf
import numpy as np

reader = tf.train.NewCheckpointReader('/home/zl198/github/users/666zcli/models/tutorials/image/cifar10/Adam_finetune_bias_tuning_lr_0.0001_ti_150000_ellipse_weight_decay_0.015/cifar10_train/model.ckpt-150000')
all_variables = reader.get_variable_to_shape_map()
w1 = reader.get_tensor("conv1/weights")
print(type(w1))
print(w1.shape)
print(w1[0])
