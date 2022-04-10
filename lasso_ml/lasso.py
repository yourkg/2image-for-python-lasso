'''/*
2022.3.21 v,1 read and save .csv
2022.3.23 v,2 lasso
2022.3.24 v,3 plt figure show
2022.3.30 画图微调
*/'''

import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV


'''# 表格处理部分
import os
file_list = os.listdir('./data/')

i=0

for file_name in file_list:
    print(file_name.split('.')[0],i)
    #读取，如果读取.csv用pd.read_csv,这里我用excel测试的
    this_data=pd.read_excel('./data/'+file_name)
    #截取
    this_data=this_data.iloc[37:,2:] 
    #增加姓名行
    top_row = pd.DataFrame({'Feature Name':['label']})
    this_data = pd.concat([top_row, this_data]).reset_index(drop = True)
    this_data.iloc[0,1:]=file_name.split('.')[0]
    #转置
    this_data=pd.DataFrame(this_data.values.T)#,columns=this_data['Feature Name'])
    #print(this_data)
    #拼接
    if i==0:
        combined_data=this_data
    else:
        #print(this_data.iloc[1:,:])
        combined_data=pd.concat([combined_data,this_data.iloc[1:,:]]).reset_index(drop = True)
    i+=1
print(combined_data)
combined_data.to_csv('./new_data.csv',index=0,header=0)
'''

#lasso 部分 对应那个视频
#先从表格读取数据
data=pd.read_csv('./new_data.csv')
x=data[data.columns[1:]]
y=data['label']
colNames=x.columns

#规整化x
x=x.astype(np.float64)
x=StandardScaler().fit_transform(x)
x=pd.DataFrame(x)
x.columns=colNames
x_raw=x
#调用sklearn的lasso方法，对x，y建模
'''
这里有几个参数，具体用法需要知道lasso算法的具体步骤
我们先用视频中的参数试一下，之后您进行调参，需要具体了解lasso
因为我们的样本不够多，这里我做了稍微调整

'''
model=LassoCV(alphas=np.logspace(-3,1,200),cv=2,max_iter=10000).fit(x,y)

#建模之后，我们可以在model里取得我们需要的信息
print(model.alpha_)
coef=pd.Series(model.coef_,index=x.columns)
print("Lasso picked \n"
      +str(sum(coef!=0))
      +"\n variables and eliminated the other \n"
      +str(sum(coef==0)))

#输出降维后的特征
index = coef[coef !=0].index
x=x[index]
print(coef[coef !=0])






#修改后的画图
#修改后的画图
#修改后的画图
#修改后的画图
#修改后的画图
#修改后的画图
#修改后的画图
#修改后的画图
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


def draw2(model,a1,a2,a3,max_iter=10000):
    """
    输入:model,lasso模型,a1,a2,a3,分别为logspace三个参数
    """
    coefs=model.path(x_raw,y,alphas=np.logspace(a1,a2,a3),max_iter=max_iter)[1].T
    fig,ax_f=plt.subplots()
    plt.xlim(10**a1,10**a2)
    ax_c=ax_f.twiny()
    ax_f.semilogx(model.alphas_,coefs,'-')
    ax_f.axvline(model.alpha_,color='black',ls='--')
    ax_f.set_xlabel('Lambda')
    ax_f.set_ylabel('Coefficients')
    
    ncoefs=coefs.copy()
    ncoefs[ncoefs!=0]=1
    xsum=ncoefs.sum(axis=1)

    #print(list(range(a3))[::40],xsum[::40])
    k=a3//(a2-a1+1)
    ax_c.set_xticks(list(range(a3))[::k],labels=list(map(int,xsum))[::k])
    ax_c.invert_xaxis()
    plt.show()

    MSEs=(model.mse_path_)
    MSEs_mean=np.apply_along_axis(np.mean,1,MSEs)
    MSEs_std=np.apply_along_axis(np.std,1,MSEs)

    fig,ax_f=plt.subplots()
    plt.xlim(10**a1,10**a2)
    ax_c=ax_f.twiny()
    k=a3//20
    ax_c.set_xticks(list(range(a3))[::10],labels=list(map(int,xsum))[::10])
    ax_c.invert_xaxis()
    ax_f.errorbar(model.alphas_,MSEs_mean
                 ,yerr=MSEs_std
                 ,fmt="o"
                 ,ms=1
                 ,mfc="r"
                 ,mec="r"
                 ,ecolor="lightblue"
                 ,elinewidth=1
                 ,capsize=3
                 ,capthick=1)
    ax_f.semilogx()

    ax_f.axvline(model.alpha_,color='black',ls='--')
    ax_f.set_xlabel('Lambda')
    ax_f.set_ylabel('*Binomial Deviance')
    ax=plt.gca()
    y_major_locator=MultipleLocator(0.06)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.show()

draw2(model,-3,1,200,max_iter=10000)












