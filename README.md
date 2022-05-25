# EX-06-Feature-Transformation

## AIM
To Perform the various feature transformation techniques on a dataset and save the data to a file. 

# Explanation
Feature Transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

 
# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature Transformation techniques to all the feature of the data set
### STEP 4
Save the data to the file


# CODE

```
NAME:RAKSHITHA DEVI J
REG NO:212221230082
```

## titanic dataset
```
import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

df=pd.read_csv("titanic_dataset.csv")
df.info()

df.isnull().sum()

df['Cabin']=df['Cabin'].fillna(df['Cabin'].mode()[0])
df['Age']=df['Age'].fillna(df['Age'].mean())
df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])
df.isnull().sum()

df.skew()
df1=df.copy()
df1=df.info()
df1.skew()
df1["Sibsp_1"]=np.sqrt(df1.SibSp)
df1.SibSp.hist()
df1.skew()
df

del df['Name']
df

del df['Cabin']
del df['Ticket']
df.isnull().sum()

from sklearn.preprocessing import
OrdinalEncoder
embark=["C","S","Q"]
emb=OrdinalEncoder (categories =[embark])
df["Embarked"]=emb.fit_transform(df[["Embarked"]])
df

from category_encoders import BinaryEncoder
be1=BinaryEncoder()
df['Sex']=be1.fit_transform(df[["Sex"]])
df


#Function Transformation:
#Log Tranformation:
np.log(df["Age"])

#Reciprocal Transformation
np.reciprocal (df[["Fare"]])

#sqrt transformation
np.sqrt(df["Embarked"])

#power transformation
df["Age_boxcox"],parameters=stats.boxcox(df["Age"])
df


df["Pclass_boxcox"],parameters=stats.boxcox(df["Pclass"])
df

df["Fare_yeojohnson"],parameters = stats.yeojohnson(df["Fare"])
df

df["Parch_yeojohnson"],parameters = stats.yeojohnson(df["Parch"])
df

df.skew()

#Quantile transformation

from sklearn.preprocessing import QuantileTransformer

qt=QuantileTransformer(output_distribution ='normal',n_quantiles=891)

df["Age_1"]=qt.fit_transform(df[["Age"]])
sm.qqplot(df['Age'],line='45')

sm.qqplot(df['Age_1'],line='45')

df["Fare_1"]=qt.fit_transform(df[["Fare"]])
sm.qqplot(df["Fare"],line='45')
sm.qqplot(df['Fare_1'],line='45')

df["Parch_1"]=qt.fit_transform(df[["Parch"]])
sm.qqplot(df['Parch'],line='45')
sm.qqplot(df['Parch_1'],line='45')

df
```
## data transform
```
import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

df=pd.read_csv("Data_To_Transform.csv")
df

df.skew()

#Function Transformation 
#Log Transformation 
np.log(df["Highly Positive Skew"])
np.reciprocal(df["Moderate Positive Skew"])
np.sqrt(df["Highly Positive Skew"])

df["Highly positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])

df["Moderate Positive Skew_yeojohnson"],parameters=stats.boxcox(df["Moderate Positive Skew"])
df

df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df


df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df

df.skew()
#Quantile Transformation 
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution ='normal')

df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')

df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')

df["Highly Positive Skew_1"]=qt.fit_transform(df[["Highly Positive Skew"]])
sm.qqplot(df["Highly Positive Skew"],line='45')

df
```

# OUPUT
## output for titanic dataset
![image](https://user-images.githubusercontent.com/94165326/170322433-bfc0580f-4cd3-417c-857b-c4ce5b2d445b.png)
![image](https://user-images.githubusercontent.com/94165326/170322465-f8d9dd5a-548e-4316-866e-a7cd748b89e6.png)
![image](https://user-images.githubusercontent.com/94165326/170322536-f329b79b-e9db-43af-9cc8-ea7ad4d55e93.png)
![image](https://user-images.githubusercontent.com/94165326/170322574-b685c7f8-4ed0-4411-8db7-1648b3e61d7b.png)
![image](https://user-images.githubusercontent.com/94165326/170322615-556256f5-7e6d-412f-a721-8e1a86bcbfed.png)
![image](https://user-images.githubusercontent.com/94165326/170322658-81989a2f-febd-42cc-ab3d-464c71dca26c.png)
![image](https://user-images.githubusercontent.com/94165326/170322690-ee18a7d8-75af-4146-b04b-8a883928dd15.png)
![image](https://user-images.githubusercontent.com/94165326/170322732-64de35a5-dac9-4960-beed-85aa9c515552.png)
![image](https://user-images.githubusercontent.com/94165326/170322788-d0cb9528-73a4-4100-af77-6f74da5ecd89.png)
![image](https://user-images.githubusercontent.com/94165326/170322817-67f2f1a4-5daa-4626-ba20-81a4d08a26c4.png)
![image](https://user-images.githubusercontent.com/94165326/170322851-7d318479-ccf1-46d6-8f32-7fbb17aee9ba.png)
![image](https://user-images.githubusercontent.com/94165326/170323294-7ce0e894-0c7a-417c-8666-ae576260d6bd.png)

## output for data transform
![image](https://user-images.githubusercontent.com/94165326/170323647-11d77822-9e59-43b4-8cf4-5d21e245748d.png)
![image](https://user-images.githubusercontent.com/94165326/170323674-94650373-3136-4c47-8565-1ab61191f853.png)
![image](https://user-images.githubusercontent.com/94165326/170323717-8e892ccb-520f-40de-9fbb-09a70c8cffc2.png)
![image](https://user-images.githubusercontent.com/94165326/170323798-eebf2eba-c138-4d5f-a8fa-e2b3fb91ce62.png)
![image](https://user-images.githubusercontent.com/94165326/170323834-9a674b5b-0543-44ac-bade-cd248c3bfdee.png)
![image](https://user-images.githubusercontent.com/94165326/170323892-fe7ae8b4-8c9c-46d8-893b-fa15af56b67d.png)
![image](https://user-images.githubusercontent.com/94165326/170323935-d0ceb154-9933-499f-9db2-250737d74ee5.png)
![image](https://user-images.githubusercontent.com/94165326/170323971-3bade022-ec75-4ab5-9b27-5df05da62775.png)
![image](https://user-images.githubusercontent.com/94165326/170324012-3c0d852d-f964-4627-8abc-254cb2d9278b.png)
![image](https://user-images.githubusercontent.com/94165326/170324065-af76b5b3-b80a-4d07-a44c-9fff4cba69c6.png)
![image](https://user-images.githubusercontent.com/94165326/170324091-806f8c8e-d574-4343-8c4f-4637b03dfb47.png)
![image](https://user-images.githubusercontent.com/94165326/170324120-7d5c7243-1703-4f72-a3f9-9bb377c3bfd2.png)
![image](https://user-images.githubusercontent.com/94165326/170324148-1b55bff3-56dc-4d15-b129-42b158ae3904.png)
![image](https://user-images.githubusercontent.com/94165326/170324262-9a0583cf-4f88-42e0-878e-ccb13c4a9a90.png)
![image](https://user-images.githubusercontent.com/94165326/170324292-4c26897e-b81a-4159-a836-0dd9bc703de6.png)

## Result:
The various feature transformation techniques has been performed on the given datasets and the data are saved to a file.





