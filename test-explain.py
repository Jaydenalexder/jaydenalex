import numpy as np
import tensorflow as tf



from tensorflow.keras.layers import Add, Conv2D, Input, Lambda
from tensorflow.keras.models import Model



DIV2K_RGB_MEAN = np.array([0.4488, 0.4371, 0.4040]) * 255




def edsr(scale, num_filters=64, num_res_blocks=8, res_block_scaling=None):#放大尺度 过滤器数量 残差块数量 残差块缩放
    #创建EDSR  https://blog.csdn.net/hai_john/article/details/102969772
    x_in = Input(shape=(None, None, 3))#由于高宽的输入未指定，所以其张量形状为（None，None，None，256）
    x = Lambda(normalize)(x_in)#匿名函数 标准化/正规化



    x = b = Conv2D(num_filters, 3, padding='same')(x)#二维卷积层 滤波器数量64 内核数3  same填充
    for i in range(num_res_blocks):
        b = res_block(b, num_filters, res_block_scaling)
    b = Conv2D(num_filters, 3, padding='same')(b)  #https://blog.csdn.net/koala_cola/article/details/106883961
    x = Add()([x, b])
    
    
    

    x = upsample(x, scale, num_filters)#扩大特征图的方法，upsample/上采样的方法
    x = Conv2D(3, 3, padding='same')(x)

    x = Lambda(denormalize)(x)#匿名函数 非标准化/非正规化
    return Model(x_in, x, name="edsr")




def res_block(x_in, filters, scaling):
    #建立EDSR残差剩余块
    x = Conv2D(filters, 3, padding='same', activation='relu')(x_in)
    x = Conv2D(filters, 3, padding='same')(x)
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
    x = Add()([x_in, x])
    return x




def upsample(x, scale, num_filters):#扩大特征图的方法，upsample/上采样的方法
    def upsample_1(x, factor, **kwargs):
        
        #子像素卷积
        x = Conv2D(num_filters * (factor ** 2), 3, padding='same', **kwargs)(x)
        return Lambda(pixel_shuffle(scale=factor))(x) #分辨率增大 #https://blog.csdn.net/g11d111/article/details/82855946



    if scale == 2:
        x = upsample_1(x, 2, name='conv2d_1_scale_2')
    elif scale == 3:
        x = upsample_1(x, 3, name='conv2d_1_scale_3')
    elif scale == 4:
        x = upsample_1(x, 2, name='conv2d_1_scale_2')
        x = upsample_1(x, 2, name='conv2d_2_scale_2')

    return x




def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)
                    #从深度大小为 block_size * block_size 的数据块重新排列成不重叠的大小为 block_size x block_size 的数据块.
                    #输出张量的宽度为 input_depth * block_size,而高度是 input_height * block_size.





def normalize(x): #正规化函数
    return (x - DIV2K_RGB_MEAN) / 127.5




def denormalize(x): #非正规化函数
    return x * 127.5 + DIV2K_RGB_MEAN


from data import DIV2K
#下载DIV2K_trian_LR_bicubic_X2.zip
train = DIV2K(scale=4, downgrade='bicubic', subset='train')
#train = DIV2K(scale=2, downgrade='bicubic', subset='train')
train_ds = train.dataset(batch_size=16, random_transform=True)



import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay



# Create directory for saving model weights
#创建用于保存模型权重的目录
weights_dir = 'weights/article'
os.makedirs(weights_dir, exist_ok=True)



# EDSR baseline as described in the EDSR paper (1.52M parameters)
#EDSR文件中描述的EDSR基线(1.52M参数)  
model_edsr = edsr(scale=4, num_res_blocks=16)



# Adam optimizer with a scheduler that halfs learning rate after 200,000 steps
#使用Adam优化器200,000步后学习速度减半  
optim_edsr = Adam(learning_rate=PiecewiseConstantDecay(boundaries=[200], values=[1e-4, 5e-5]))



# Compile and train model for 300,000 steps with L1 pixel loss
#对L1像素损失收集和训练模型300,000步
model_edsr.compile(optimizer=optim_edsr, loss='mean_absolute_error')
model_edsr.fit(train_ds, epochs=3, steps_per_epoch=10)

# Save model weights
#保存模型权重
model_edsr.save_weights(os.path.join(weights_dir, 'weights-edsr-16-x4.h5'))



from model import srgan


#使用内容损失  /均方误差
mean_squared_error = tf.keras.losses.MeanSquaredError()#均方差损失函数


#使用生成器损失和判别器损失
binary_cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)#计算真实标签和预测标签之间的交叉熵损失
#from_logits
#是否将y_pred解释为logit值的张量。
#默认情况下，我们假设y_pred包含概率（即[0，1]中的值



##在VGG19中计算第5个max-pooling层之前的第4次卷积之后的feature map的模型。这是相应Keras模型中的第20层。
vgg = srgan.vgg_54()




#使用EDSR模型作为SRGAN的生成器
generator = edsr(scale=4, num_res_blocks=16)#放大尺度 残差块数量
generator.load_weights(os.path.join(weights_dir, 'weights-edsr-16-x4.h5'))



# SRGAN discriminator
#SRGAN判别器
discriminator = srgan.discriminator()



#优化生成器和判别器。SRGAN训练200,000步，100,000步后学习率由1e-4降至1e-5
schedule = PiecewiseConstantDecay(boundaries=[100000], values=[1e-4, 1e-5])#分段函数 边界100000
generator_optimizer = Adam(learning_rate=schedule)
discriminator_optimizer = Adam(learning_rate=schedule)



def generator_loss(sr_out):
    return binary_cross_entropy(tf.ones_like(sr_out), sr_out) #生成一个与sr_out类型相同但值为1的，赋值给函数binary_cross_entropy



def discriminator_loss(hr_out, sr_out):
    hr_loss = binary_cross_entropy(tf.ones_like(hr_out), hr_out)#生成一个与hr_out类型相同但值为1的，赋值给函数binary_cross_entropy
    sr_loss = binary_cross_entropy(tf.zeros_like(sr_out), sr_out)##生成一个与sr_out类型相同但值为1的，赋值给函数binary_cross_entropy
    return hr_loss + sr_loss



@tf.function  #实现Graph Execution，从而将模型转换为易于部署且高性能的 TensorFlow 图模型
def content_loss(hr, sr):
    sr = tf.keras.applications.vgg19.preprocess_input(sr)
    #函数https://vimsky.com/zh-tw/examples/usage/python-tf.keras.applications.vgg19.preprocess_input-tf.html
    '''
    参数：
    sr具有 3 個顏色通道的浮點 其值在 [0, 255] 範圍內。
    如果数据类型兼容,则预处理数据将覆盖输入数据。为了避免这一行为,可以使用numpy.copy(sr)。
    data_format 图像张量/数组的可选数据格式，默认为无、
    返回值：
    图像从RGB转换为BGR，然后每个颜色通道相对于imageNrt数据集为zero-centered，无需缩放
    预处理编码一批图像的张量或numpy数组
    '''
    hr = tf.keras.applications.vgg19.preprocess_input(hr)
    sr_features = vgg(sr) / 12.75 #调用vgg函数 line54
    hr_features = vgg(hr) / 12.75
    return mean_squared_error(hr_features, sr_features) #返回值为计算hr、lr的均方误差



@tf.function
def train_step(lr, hr):
#SRGAN训练步骤。
#将LR和HR图像批处理作为输入和返回
#计算感知损失和鉴别器损失。
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:  #构建梯度环境-生成器梯度和判别器梯度
        lr = tf.cast(lr, tf.float32)#将lr的类型转换为float32型
        hr = tf.cast(hr, tf.float32)



        # Forward pass
        # #正向传播
        sr = generator(lr, training=True)
        hr_output = discriminator(hr, training=True)
        sr_output = discriminator(sr, training=True)



        #计算损失
        con_loss = content_loss(hr, sr) #内容损失
        gen_loss = generator_loss(sr_output) #生成器损失
        perc_loss = con_loss + 0.001 * gen_loss  #感知损失
        disc_loss = discriminator_loss(hr_output, sr_output) #判别器损失



    # Compute gradient of perceptual loss w.r.t. generator weights 
    #计算感知损失梯度w.r.t. 生成器权重 
    gradients_of_generator = gen_tape.gradient(perc_loss, generator.trainable_variables) #gen_tape.gradient生成器梯度
    # Compute gradient of discriminator loss w.r.t. discriminator weights 
    #计算判别器损失梯度w.r.t.鉴别器权重
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)



    # Update weights of generator and discriminator
    #更新生成器和判别器的权重
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


    return perc_loss, disc_loss



pls_metric = tf.keras.metrics.Mean()#计算给定值的（加权）平均值 生成损失
dls_metric = tf.keras.metrics.Mean()#                         判别损失



steps = 200000
step = 0



# Train SRGAN for 200,000 steps.
#以200000步训练SRGAN
for lr, hr in train_ds.take(steps):#用train_ds.take(每步)方式取出的是批量图片张量，张量可用序号
    step += 1

    pl, dl = train_step(lr, hr) #调用train_step函数
    pls_metric(pl)#调用函数传参
    dls_metric(dl)#调用函数传参

    if step % 50 == 0:#判断步长是否小于一半 小于一半学习率改变
        print(f'{step}/{steps}, perceptual loss = {pls_metric.result():.4f}, discriminator loss = {dls_metric.result():.4f}')
        pls_metric.reset_states() #reset_states,删除内部状态 //清楚网络隐藏状态
        dls_metric.reset_states()
        
        
        
generator.save_weights(os.path.join(weights_dir, 'weights-edsr-16-x4-fine-tuned.h5'))#保存调整完成的权重





import os
import matplotlib
import matplotlib.pyplot as plt

from model import resolve_single  #common.py里
from utils import load_image

#%matplotlib inline



def resolve_and_plot(model_pre_trained, model_fine_tuned, lr_image_path):
    lr = load_image(lr_image_path) #加载图像
    
    sr_pt = resolve_single(model_pre_trained, lr)  #训练前图像 #common里的函数
    sr_ft = resolve_single(model_fine_tuned, lr)   #训练后图像 #common里的函数
    
    plt.figure(figsize=(20, 20)) #设置图像长宽为20英寸
    
    model_name = model_pre_trained.name.upper() #name.upper()#将字符串中的每个单词全部字母都大写
    images = [lr, sr_pt, sr_ft]
    titles = ['LR', f'SR ({model_name}, pixel loss)', f'SR ({model_name}, perceptual loss)']
    positions = [1, 3, 4]
    
    for i, (image, title, position) in enumerate(zip(images, titles, positions)):
        plt.subplot(2, 2, position)
        plt.imshow(image)
        plt.title(title)
        plt.xticks([])
        plt.yticks([])
     
     
        
weights_dir = 'weights/article'



edsr_pre_trained = edsr(scale=4, num_res_blocks=16)
edsr_pre_trained.load_weights(os.path.join(weights_dir, 'weights-edsr-16-x4.h5'))



edsr_fine_tuned = edsr(scale=4, num_res_blocks=16)
edsr_fine_tuned.load_weights(os.path.join(weights_dir, 'weights-edsr-16-x4-fine-tuned.h5'))



#resolve_and_plot(edsr_pre_trained, edsr_fine_tuned, 'demo/0869x4-crop.png')
resolve_and_plot(edsr_pre_trained, edsr_fine_tuned, 'demo/hua.jpg')
#resolve_and_plot(edsr_pre_trained, edsr_fine_tuned, 'demo/maomao3.jpg') #error
plt.show()




from model.wdsr import wdsr_b



wdsr_pre_trained = wdsr_b(scale=4, num_res_blocks=32)
wdsr_pre_trained.load_weights(os.path.join(weights_dir, 'weights-wdsr-b-32-x4.h5'))



wdsr_fine_tuned = wdsr_b(scale=4, num_res_blocks=32)
wdsr_fine_tuned.load_weights(os.path.join(weights_dir, 'weights-wdsr-b-32-x4-fine-tuned.h5'))



resolve_and_plot(wdsr_pre_trained, wdsr_fine_tuned, 'demo/0829x4-crop.png')
plt.show()