print('*'*80)
print('.......................Welcome to Sandeep MLOps task o1...........................')
import pandas as pd 
from sklearn.linear_model import LinearRegression
print('*'*80)
db=pd.read_csv('test_data_set.csv')
print('We have a Data Set Now we are used to and now we are  show  ...')
print('#'*80)
print(db)
print('#'*80)
#type(db)
y= db["mark"]
x= db["NumberOfStrd"]
x.shape
x=x.values
x = x.reshape(8,1)
x.shape
mind = LinearRegression()
mind.fit(x,y)

a=int(input('enter your value : -'))
b=mind.predict([[a]])
print('#'*80)
print('................ Your Resut is...................')
print('#'*80)
print('predict data :- ',b)
print('#'*80)
print('.......................... Thank you for Using App........')
print('#'*80)
