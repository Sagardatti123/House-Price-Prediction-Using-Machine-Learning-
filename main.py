
''' Ask the user for input values and predict the price using the trained model.                       '''
import sys
import pickle
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,root_mean_squared_error


class MULTI_REGRESSION:# class name

    def __init__(self):#initilize the constructor
        try:
            self.df=pd.read_csv('data.csv')
        except Exception as e:
            er_type,er_msg,er_line=sys.exc_info()
            print(er_type,er_msg,er_line.tb_lineno)



    def clean_data(self):
        try:
            self.df['date']=pd.to_datetime(self.df['date'])#to_datetime function used divide the day,month,year
            self.df['day']=self.df['date'].dt.day
            self.df['month']=self.df['date'].dt.month
            self.df['year']=self.df['date'].dt.year
            self.df=self.df.drop(columns=['date'])#delete the date column
            self.df['city']=self.df['city'].astype('category').cat.codes#here cat.codes is change the categorycal values into numerical values
            self.df['country']=self.df['country'].astype('category').cat.codes
            print(f'data:{self.df.head(5)}')
        except Exception as e:
            er_type,er_msg,er_line=sys.exc_info()
            print(er_type,er_msg,er_line.tb_lineno)

    def split_data(self):
        try:
            self.X=self.df.iloc[ : ,1:]
            self.y=self.df.iloc[ : ,0]
            self.X_train,self.X_test,self.y_train,self.y_test=train_test_split(self.X,self.y,test_size=0.2,random_state=42)#divide the training_data and testing_data
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(er_type, er_msg, er_line.tb_lineno)
    def training_data(self):
        try:
            self.obj=LinearRegression()#it will learns
            self.obj.fit(self.X_train,self.y_train)# gives predictions in the form of numericals
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(er_type, er_msg, er_line.tb_lineno)

    def train_accuracy_rmse(self):
            try:
                self.train_pre=self.obj.predict(self.X_train)

                print('+--------------+')
                #from sklearn.metrics import r2_score,root_mean_squared_error
                #finding the accuracy
                y_train_mean=self.y_train.mean()
                numerator1=((self.y_train-self.train_pre)**2).sum()
                denomaretor1=((self.y_train-y_train_mean)**2).sum()
                r2=1-(numerator1/denomaretor1)
                print(f'r2_score_train:{r2}')
                print(f'r2_score_train:{r2_score(self.y_train,self.train_pre)}')
                print('+---------------+')

                #finding the root mean squred error

                numerator2=((self.y_train-self.train_pre)**2).sum()
                denomaretor2=len(self.y_train)
                rmse=np.sqrt(numerator2/denomaretor2)
                print(f'rmse:{rmse}')
                print(f'rmse:{root_mean_squared_error(self.y_train,self.train_pre)}')
                print('+------------+')
            except Exception as e:
                er_type, er_msg, er_line = sys.exc_info()
                print(er_type, er_msg, er_line.tb_lineno)
    def test_accuracy_rmse(self):
        try:
            test_pre=self.obj.predict(self.X_test)
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
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(er_type, er_msg, er_line.tb_lineno)

    def coef_inter_(self):
        try:
            print(f'M:{self.obj.coef_}')
            print(f'C:{self.obj.intercept_}')
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(er_type, er_msg, er_line.tb_lineno)

    def pickle(self):
        try: #save the model
            with open('houseprice.pkl','wb') as a:
                pickle.dump(self.obj,a)
            print('model save successfully')
            #load the model
            with open('houseprice.pkl','rb') as f:
                self.model=pickle.load(f)
            print('model loaded successfully')
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(er_type, er_msg, er_line.tb_lineno)
    def user_input(self):
        try:
            bedrooms=int(input('enter no of bedrooms:'))
            bathrooms=float(input('enter no of bathrooms'))
            sqft_living = float(input("Enter living area (sqft): "))
            sqft_lot = float(input("Enter lot area (sqft): "))
            floors = float(input("Enter number of floors: "))
            waterfront = int(input("Enter waterfront: "))
            view = int(input("Enter view rating: "))
            condition = int(input("Enter condition: "))
            sqft_above = float(input("Enter sqft above: "))
            sqft_basement = float(input("Enter sqft basement: "))
            yr_built = int(input("Enter year built: "))
            yr_renovated = int(input("Enter year renovated:"))
            city = int(input("Enter city code: "))
            country = int(input("Enter country code: "))
            day = int(input("Enter day: "))
            month = int(input("Enter month: "))
            year = int(input("Enter year: "))
            user_data = pd.DataFrame({'bedrooms': [bedrooms],'bathrooms': [bathrooms],
                                      'sqft_living': [sqft_living],'sqft_lot': [sqft_lot],
                                      'floors': [floors],'waterfront': [waterfront],
                                      'view': [view],'condition': [condition],
                                      'sqft_above': [sqft_above],'sqft_basement': [sqft_basement],
                                      'yr_built': [yr_built],'yr_renovated': [yr_renovated],
                                      'city': [city],'country': [country],'day': [day],
                                      'month': [month],'year': [year]})
            prediction=self.model.predict(user_data)
            print(f'house price in dollors:{prediction[0]:,.2f}')
            print(f'house price in ruppes:{prediction[0]*85:,.2f}')

        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(er_type, er_msg, er_line.tb_lineno)
if __name__=='__main__':
    try:
        object=MULTI_REGRESSION()#calling constractor
        object.clean_data()
        object.split_data()
        object.training_data()
        object.train_accuracy_rmse()
        object.test_accuracy_rmse()
        object.coef_inter_()
        object.pickle()
        object.user_input()
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        print(er_type, er_msg, er_line.tb_lineno)


