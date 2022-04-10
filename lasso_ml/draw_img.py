
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV


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












