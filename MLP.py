import d2lzh as d2l
from mxnet import nd,gluon,init
from mxnet.gluon import loss as gloss, nn
'''
#读取数据集
batch_size = 256
train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size) #图像为28*28 类别为10

#定义模型参数
num_inputs,num_outputs,num_hiddens = 784,10,256 #输入、输出、隐藏层各unit数量
W1 = nd.random.normal(scale=0.01,shape=(num_inputs,num_hiddens))
b1 = nd.zeros(num_hiddens)
W2 = nd.random.normal(scale=0.01,shape=(num_hiddens,num_outputs))
b2 = nd.zeros(num_outputs)
params = [W1,b1,W2,b2]
for param in params:
    param.attach_grad()#申请相应内存来存放参数的导数，后期使用backgrad自动求导

#定义激活函数
def relu(X):
    return nd.maximum(0,X)
#定义模型
def net(X):
    X = X.reshape((-1,num_inputs))
    H = relu(nd.dot(X,W1) + b1)
    return nd.dot(H,W2) + b2
#定义损失函数
loss = gloss.SoftmaxCrossEntropyLoss()

#训练模型
num_epochs,lr = 5,0.5
d2l.train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,params,lr)#lr为学习率
'''

'''以下为简洁实现版本'''
net = nn.Sequential()#模型实例化
net.add(nn.Dense(256,activation='relu'),nn.Dense(10))#添加一隐藏层和一输出层
net.initialize(init.Normal(sigma=0.01))#参数随机初始化，均值为0，标准差为0.01
batch_size = 256
train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size) #图像为28*28 类别为10
loss = gloss.SoftmaxCrossEntropyLoss()#损失函数定义
trainer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':0.5})#训练器设定
num_epochs = 5
d2l.train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,None,None,trainer)#训练