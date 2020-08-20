from mxnet import autograd,nd
import d2lzh as d2l
from mxnet import gluon,init
from mxnet.gluon import loss as gloss,nn

#读取数据集
batch_size = 256
train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size)
#可视化训练数据集中前20个图像
#X,y = train_iter[0:9]
#d2l.show_fashion_mnist(X,d2l.get_fashion_mnist_labels(y))
#d2l.plt.show()
#定义和初始化模型
net = nn.Sequential()#模型实例化
net.add(nn.Dense(10))#输出层输出个数为10
net.initialize(init.Normal(sigma=0.01))#用均值为0、标准差为0.01初始化权重参数
#softmax运算与交叉熵损失函数
loss = gloss.SoftmaxCrossEntropyLoss()
#定义优化算法
trainer = gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':0.1})
#训练模型
num_epochs = 5
d2l.train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,None,None,trainer)

