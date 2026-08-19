import sys
from asyncio import exceptions
from copyreg import pickle

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,root_mean_squared_error

try:
    class regression:
        def __init__(self):
            self.df=pd.read_csv('data.csv')


        def clean_data(self):
            self.df['date']=pd.to_datetime(self.df['date'])
            self.df['day']=self.df['date'].dt.day
            self.df['month']=self.df['date'].dt.month
            self.df['year']=self.df['date'].dt.year
            self.df=self.df.drop(columns=['date'])
            self.df['city']=self.df['city'].astype('category').cat.codes
            self.df['country']=self.df['country'].astype('category').cat.codes
            print(f'data:{self.df.head(5)}')


        def split_data(self):
            self.X=self.df.iloc[ : ,1:]
            self.y=self.df.iloc[ : ,0]
            self.X_train,self.X_test,self.y_train,self.y_test=train_test_split(self.X,self.y,test_size=0.2,random_state=42)


        def algorithm(self):
            self.obj=LinearRegression()
            self.obj.fit(self.X_train,self.y_train)

        def train_accuracy_rmse(self):
            self.train_pre=self.obj.predict(self.X_train)
            #print(f'train_pre:{self.train_pre}')
            #r2_score
            print('+--------------+')
            from sklearn.metrics import r2_score,root_mean_squared_error
            y_train_mean=self.y_train.mean()
            numerator1=((self.y_train-self.train_pre)**2).sum()
            denomaretor1=((self.y_train-y_train_mean)**2).sum()
            r2=1-(numerator1/denomaretor1)
            print(f'r2_score_train:{r2}')
            print(f'r2_score_train:{r2_score(self.y_train,self.train_pre)}')
            print('+---------------+')

            #rmse

            numerator2=((self.y_train-self.train_pre)**2).sum()
            denomaretor2=len(self.y_train)
            rmse=np.sqrt(numerator2/denomaretor2)
            print(f'rmse:{rmse}')
            print(f'rmse:{root_mean_squared_error(self.y_train,self.train_pre)}')
            print('+------------+')

        def test_accuracy_rmse(self):
            test_pre=self.obj.predict(self.X_test)
            #print(f'test_pre:{test_pre}')
            print('+-----------+')

            #r2_score()
            y_test_mean=self.y_test.mean()
            numretor3=((self.y_test-test_pre)**2).sum()
            denomretor3=((self.y_test-y_test_mean)**2).sum()
            r2=1-(numretor3/denomretor3)
            print(f'r2_score_test:{r2}')
            print(f'r2_score_test:{r2_score(self.y_test,test_pre)}')
            print('+-----------+')
            #rmse
            numretor4=((self.y_test-test_pre)**2).sum()
            denomretor4=(len(self.y_test))
            rmse=np.sqrt(numretor4/denomretor4)
            print(f'rmse_test:{rmse}')
            print(f'rmse_test:{root_mean_squared_error(self.y_test,test_pre)}')

            print('+-----------+')

        def coef_inter_(self):
            print(f'M:{self.obj.coef_}')
            print(f'C:{self.obj.intercept_}')

    object=regression()
    object.clean_data()
    object.split_data()
    object.algorithm()
    object.train_accuracy_rmse()
    object.test_accuracy_rmse()
    object.coef_inter_()
    with open('houseprice.pkl','w') as e:
        pickle.dumps(object,e)

except Exception as e:
    er_type,er_mess,er_line=sys.exc_info()
    print(er_type)
    print(er_mess)
    print(er_line)