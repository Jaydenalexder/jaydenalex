import numpy as np
import tensorflow as tf


DIV2K_RGB_MEAN = np.array([0.4488, 0.4371, 0.4040]) * 255


def resolve_single(model, lr):
    return resolve(model, tf.expand_dims(lr, axis=0))[0]
                         #tf.expend_dims https://blog.csdn.net/sereasuesue/article/details/109011721

def resolve(model, lr_batch):
    lr_batch = tf.cast(lr_batch, tf.float32) #tf.cast 将 x 的数据格式转化成 dtype. 
                                             #例如，原来 x 的数据格式是 bool，那么将其转化成 float 以后，就能够将其转化成 0 和 1 的序列。
                                             #反之也可以
   
    sr_batch = model(lr_batch)
   
    sr_batch = tf.clip_by_value(sr_batch, 0, 255) #将一个张量中的数值限制在一个范围之内 运用的是交叉熵而不是二次代价函数
    
    sr_batch = tf.round(sr_batch) #tf.round 四舍五入为整数
   
    sr_batch = tf.cast(sr_batch, tf.uint8)  #tf.cast 将 x 的数据格式转化成 dtype. 
    
    return sr_batch


def evaluate(model, dataset): #定义评价函数
    psnr_values = []
    for lr, hr in dataset:
        sr = resolve(model, lr)
        psnr_value = psnr(hr, sr)[0]  #psnr峰值信噪比
        psnr_values.append(psnr_value) #append插入指定内容
    return tf.reduce_mean(psnr_values)


# ---------------------------------------
#  归一化
# ---------------------------------------


def normalize(x, rgb_mean=DIV2K_RGB_MEAN): #归一化
    return (x - rgb_mean) / 127.5


def denormalize(x, rgb_mean=DIV2K_RGB_MEAN): #逆归一化
    return x * 127.5 + rgb_mean


def normalize_01(x):
    """Normalizes RGB images to [0, 1]."""#将RGB图像归一化为[0，1]
    return x / 255.0


def normalize_m11(x):
    """Normalizes RGB images to [-1, 1].""" #将RGB图像归一化为[-1, 1]
    return x / 127.5 - 1


def denormalize_m11(x):
    """Inverse of normalize_m11."""  #逆归一化
    return (x + 1) * 127.5


# ---------------------------------------
#  Metrics
# ---------------------------------------


def psnr(x1, x2):
    return tf.image.psnr(x1, x2, max_val=255)


# ---------------------------------------
#  See https://arxiv.org/abs/1609.05158
# ---------------------------------------


def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)


