import d2lzh as d2l
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss,nn

def dropout(X,drop_prob):
    assert 0<=drop_prob<=1#确保丢弃概率合理
    keep_prob = 1-drop_prob#保留概率
    if keep_prob==0:#若保留概率值为0则将输入全部清零
        return X.zeros_like()
    mask = nd.random.uniform(0,1,X.shape) < keep_prob#否则筛选出相应比例的输入值保留
    return mask*X/keep_prob#同时保证期望值不变

#读取数据集
num_inputs,num_outputs,num_hiddens1,num_hiddens2 = 784,10,256,256#各层单元个数
num_epochs,lr,batch_size = 10,0.5,256#迭代次数，学习率，batch大小
train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size)

#参数初始化
W1 = nd.random.normal(scale=0.01,shape=(num_inputs,num_hiddens1))
b1 = nd.zeros(num_hiddens1)
W2 = nd.random.normal(scale=0.01,shape=(num_hiddens1,num_hiddens2))
b2 = nd.zeros(num_hiddens2)
W3 = nd.random.normal(scale=0.01,shape=(num_hiddens2,num_outputs))
b3 = nd.zeros(num_outputs)
params = [W1,b1,W2,b2,W3,b3]
for param in params:
    param.attach_grad()#申请内存保存梯度值
drop_prob1,drop_prob2 = 0.2,0.5#两层hl的丢弃概率

#定义模型
def net1(X):#使用dropout
    X = X.reshape((-1,num_inputs))
    H1 = (nd.dot(X,W1)+b1).relu()
    if autograd.is_training():#仅在训练模型时使用丢弃法
        H1 = dropout(H1,drop_prob1)
    H2 = (nd.dot(H1,W2)+b2).relu()
    if autograd.is_training():
        H2 = dropout(H2,drop_prob2)
    return nd.dot(H2,W3)+b3
loss = gloss.SoftmaxCrossEntropyLoss()

def net2(X):#不使用dropout
    X = X.reshape((-1,num_inputs))
    H1 = (nd.dot(X,W1)+b1).relu()
    H2 = (nd.dot(H1,W2)+b2).relu()
    return nd.dot(H2,W3)+b3

#训练和测试模型
print("With Dropout:")
d2l.train_ch3(net1,train_iter,test_iter,loss,num_epochs,batch_size,params,lr)
print("Without Dropout:")
d2l.train_ch3(net2,train_iter,test_iter,loss,num_epochs,batch_size,params,lr)
