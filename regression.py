                
              
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, SGDRegressor, Ridge
import numpy as np
import pylab as pl
from sklearn.datasets import load_boston

boston = load_boston()
print(dir(boston))
                    # ['DESCR', 'data', 'feature_names', 'filename', 'target']

print(boston.feature_names) #print the column names
                    #     ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
                    #  'B' 'LSTAT']
    
    
print(boston.data.shape) 
                    # (506, 13)
print(boston.target.shape) 
                    # (506,)
  
np.set_printoptions(precision=2, linewidth=120, suppress=True, edgeitems=4)
                #https://numpy.org/doc/stable/reference/generated/numpy.set_printoptions.html
print(boston.data)

# in order to do multiple regression we need to add a column of 1s for x0
x = np.array([np.concatenate((v,[1])) for v in boston.data])
y = boston.target #target is the price

#print off the first 10 elements to see what i am working with
print(x[:10])



a=0.3

for name,met in [\
#for name and model in this list (i.e., its a list of tuples (which are pairs))
    ('linear regression', LinearRegression()),\
    ('lasso', Lasso(fit_intercept=True, alpha=a)),\
    ('ridge', Ridge(fit_intercept=True, alpha=a)),\
    ('elastic-net', ElasticNet(fit_intercept=True, alpha=a))\
                ]:
    
    
                                                    #### Full Data Set (no validation)
      
    met.fit(x,y) #first we fit it
    p = met.predict(x) # then we predict against it
    e=p-y #margin of error = predicted price - price
    total_error = np.dot(e,e) #root mean square of the erro
    rmse_train = np.sqrt(total_error/len(p))
    diff = y-np.mean(y)
    r2_train = 1-(total_error/np.dot(diff,diff))
    
    
                                                    #### Full Data Set (no validation) (end)
 
        
        
    #plot outputs
    %matplotlib inline
    pl.plot(p,y,'ro') #estimated vs actual
    pl.plot([0,50],[0,50],'g') 
    pl.xlabel('predicted')
    pl.ylabel('predicted')
    pl.show()

    
    
                                                            #do some cross validation    
    kf =KFold(n_splits=10) #10 split
    err=0 # creating the variable for the accumulated error 
    r2_total = 0
    for train,test in kf.split(x): #for each split, 
        met.fit(x[train],y[train]) #create the moidel and do the training
        p=met.predict(x[test])
        e=p-y[test]
        diff = y[test]-np.mean(y[test])
        err += np.dot(e,e) # we accumulate the error this time (+=) 
                        #.Dot product of two arrays x and y
        
    rmse_10cv = np.sqrt(err/len(x))
    print('Method: %s' %name)
    print('RMSE on training: %.4f' %rmse_train)
    print('R2 on training: %.4f' %r2_train)
    print('RMSE on 10-fold CV: %.4f' %rmse_10cv)
    print('\n')
    
        
        
