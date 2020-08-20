import numpy as np
import pandas as pd
import d2lzh as d2l
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn

#读取数据集
train_data = pd.read_csv('E:/Machine Learning/DeepLearning/Datalib/house-prices-advanced-regression-techniques/train.csv')
test_data = pd.read_csv('E:/Machine Learning/DeepLearning/Datalib/house-prices-advanced-regression-techniques/test.csv')
print(train_data.shape,test_data.shape,"\n",train_data.iloc[0:4,[0,1,2,3,-3,-2,-1]])#查看数据规模与若干特征
all_features = pd.concat((train_data.iloc[:,1:-1],test_data.iloc[:,1:]))#去除第一个id特征，提取出训练集测试集所有特征

#预处理数据集
numeric_features = all_features.dtypes[all_features.dtypes!='object'].index#得出数值特征的索引
all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x-x.mean())/(x.std()))#标准化数值数据，减去均值除以标准差
all_features[numeric_features] = all_features[numeric_features].fillna(0)#用特征均值即0代替NA
all_features = pd.get_dummies(all_features,dummy_na=True)#将离散特征分解为指示特征
print(all_features.shape,"\n",all_features.iloc[:4,-10:])#预览转换后的数据集规模，会发现特征个数增多至331
n_train = train_data.shape[0]#训练集样本数
train_features = nd.array(all_features[:n_train].values)#最后获取训练集、测试集
test_features = nd.array(all_features[n_train:].values)
train_labels = nd.array(train_data['SalePrice'].values).reshape((-1,1))

#定义模型
def get_net():
    net = nn.Sequential()#实例化
    net.add(nn.Dense(1))#简单线性回归，1输出层
    net.initialize()#参数初始化
    return net
#定义用于评价模型的对数均方根误差
loss = gloss.L2Loss()#平方损失函数(1/2(x-x^)**2)
def log_rmse(net,features,labels):
    clipped_preds = nd.clip(net(features),1,float('inf'))#将小于1的值设为1，使得取对数时数值更稳定
    rmse = nd.sqrt(2*loss(clipped_preds.log(),labels.log()).mean())
    return rmse.asscalar()

#定义训练函数
def train(net,train_features,train_labels,test_features,test_labels,num_epochs,learning_rate,weight_decay,batch_size):
    train_ls,test_ls = [],[]
    train_iter = gdata.DataLoader(gdata.ArrayDataset(train_features,train_labels),batch_size,shuffle=True)#组合并打乱
    trainer = gluon.Trainer(net.collect_params(),'adam',{'learning_rate':learning_rate,'wd':weight_decay})#使用adam优化算法
    for epoch in range(num_epochs):
        for X,y in train_iter:
            with autograd.record():
                l = loss(net(X),y)
            l.backward()#求偏导
            trainer.step(batch_size)#更新参数
        train_ls.append(log_rmse(net,train_features,train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net,test_features,test_labels))
    return train_ls,test_ls

#K折交叉验证
def get_k_fold_data(k,i,X,y):
    assert k>1
    fold_size = X.shape[0]//k
    X_train,y_train = None,None
    for j in range(k):
        idx = slice(j*fold_size,(j+1)*fold_size)
        X_part,y_part = X[idx,:],y[idx]
        if j==i:
            X_valid,y_valid = X_part,y_part#第i折作为交叉验证集
        elif X_train is None:
            X_train,y_train = X_part,y_part
        else:
            X_train = nd.concat(X_train,X_part,dim=0)
            y_train = nd.concat(y_train,y_part,dim=0)
    return X_train,y_train,X_valid,y_valid

#依次将每一折作为交叉验证折，最后求取误差平均值
def k_fold(k,X_train,y_train,num_epochs,learning_rate,weight_decay,batch_size):
    train_l_sum,valid_l_sum = 0,0
    #分别将训练集中的每一折都作为交叉集累计误差最后求平均
    for i in range(k):
        data = get_k_fold_data(k,i,X_train,y_train)
        net = get_net()
        train_ls,valid_ls = train(net,*data,num_epochs,learning_rate,weight_decay,batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i==0:
            d2l.semilogy(range(1,num_epochs+1),train_ls,'epochs','rmse',range(1,num_epochs+1),valid_ls,['train','valid'])
        print('fold %d, train rmse %f, valid rmse %f'%(i,train_ls[-1],valid_ls[-1]))
    return train_l_sum/k,valid_l_sum/k#求取平均误差


#模型选择
k,num_epochs,lr,weight_decay,batch_size = 5,100,5,0,64
#train_l,valid_l = k_fold(k,train_features,train_labels,num_epochs,lr,weight_decay,batch_size)
#print("%d-fold validation: avg train rmse %f, avg valid rmse %f"%(k,train_l,valid_l))

#预测并在kaggle提交结果
def train_and_pred(train_features,train_labels,test_features,test_data,num_epochs,lr,weight_decay,batch_size):
    net = get_net()
    train_ls , _ = train(net,train_features,train_labels,None,None,num_epochs,lr,weight_decay,batch_size)
    d2l.semilogy(range(1,num_epochs+1),train_ls,'epochs','rmse')
    print("train rmse %f"%train_ls[-1])
    preds = net(test_features).asnumpy()
    test_data['SalePrice'] = pd.Series(preds.reshape(1,-1)[0])
    submission = pd.concat([test_data['Id'],test_data['SalePrice']],axis=1)
    submission.to_csv('submission.csv',index=False)
train_and_pred(train_features,train_labels,test_features,test_data,num_epochs,lr,weight_decay,batch_size)