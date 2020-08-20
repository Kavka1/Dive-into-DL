import d2lzh as d2l
from mxnet import autograd,gluon,init,nd
from mxnet.gluon import data as gdata, loss as gloss, nn

#构建一高维线性函数数据集
n_train,n_test,num_inputs = 100,20,200#训练样本数，测试样本数，维度/特征数
true_w,true_b = nd.ones((num_inputs,1))*0.01,0.05#真实参数

features = nd.random.normal(shape=(n_train+n_test,num_inputs))#输入数据集均值为0，方差为0.01
labels = nd.dot(features,true_w) + true_b#求得正确结果
labels += nd.random.normal(scale=0.01,shape=labels.shape)#将均值为0方差为0.01的噪声值加入结果
train_features,test_features = features[:n_train,:],features[n_train:,:]#分割训练、测试数据
train_labels,test_labels = labels[:n_train], labels[n_train:]#分割训练、测试结果

#初始化模型参数
def init_params():
    w = nd.random.normal(scale=1,shape=(num_inputs,1))
    b = nd.zeros(shape=(1,))
    w.attach_grad()
    b.attach_grad()
    return [w,b]
#定义损失函数中的L2范数惩罚项
def l2_penalty(w):
    return (w**2).sum()/2

#定义、训练、测试模型
batch_size,num_epochs,lr = 1,100,0.003#每batch仅有1个样本，迭代100次
net,loss = d2l.linreg,d2l.squared_loss#定义模型为线性回归，损失函数为差方和
train_iter = gdata.DataLoader(gdata.ArrayDataset(train_features,train_labels),batch_size,shuffle=True)#定义训练数据集,shuffle随机排列
def fit_and_plot(lambd):
    w,b = init_params()#初始换参数
    train_ls,test_ls = [],[]
    for _ in range(num_epochs):#num_epochs折
        for X,y in train_iter:
            with autograd.record():
                l = loss(net(X,w,b),y) + lambd*l2_penalty(w)#计算损失函数，注加上L2正则化惩罚项
            l.backward()#计算导数
            d2l.sgd([w,b],lr,batch_size)#梯度下降更新参数
        train_ls.append(loss(net(train_features,w,b),train_labels).mean().asscalar())#记录迭代过程中训练集的误差值
        test_ls.append(loss(net(test_features,w,b),test_labels).mean().asscalar())#记录迭代过程中测试集的误差值
    d2l.semilogy(range(1,num_epochs+1),train_ls,'epochs','loss',range(1,num_epochs+1),test_ls,['train','test'])#绘图
    print("L2 norm of w:",w.norm().asscalar())#打印w参数的范数值
fit_and_plot(lambd=8)
