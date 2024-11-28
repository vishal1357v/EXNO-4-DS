# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
        from google.colab import drive
drive.mount('/content/drive')

ls drive/MyDrive/DATA/

import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('/content/drive/My Drive/Colab Notebooks/bmi.csv')

df.head()


![image](https://github.com/user-attachments/assets/4a068c81-cad8-4828-99d4-70b9eb8c6132)


df.dropna()



![image](https://github.com/user-attachments/assets/5e4241d2-6f9d-42da-a509-75d046475f20)


max_vals = np.max(np.abs(df[["Height","Weight"]]))
max_vals



![image](https://github.com/user-attachments/assets/d77d330a-59d5-48b1-98ad-784e4fdea6d5)


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df[["Height","Weight"]]=sc.fit_transform(df[["Height","Weight"]])
df.head(10)


![image](https://github.com/user-attachments/assets/f72c8735-a0b3-4c47-8639-8f9e7bc34005)


from sklearn.preprocessing import MinMaxScaler

scalar = MinMaxScaler()
df[["Height",'Weight']]= scalar.fit_transform(df[['Height','Weight']])

df.head(10)



![image](https://github.com/user-attachments/assets/dc0eafa2-8d2a-484a-a2de-09dc1c6d36d3)


from sklearn.preprocessing import Normalizer
scalar = Normalizer()
df[['Height','Weight']]= scalar.fit_transform(df[['Height','Weight']])

df

![image](https://github.com/user-attachments/assets/33d8143f-3d08-42f3-af37-802780aba81e)


from sklearn.preprocessing import RobustScaler
scalar = RobustScaler()
df[['Height','Weight']]= scalar.fit_transform(df[['Height','Weight']])

df

![image](https://github.com/user-attachments/assets/6e8ad33d-d4bd-40d7-85b7-4887eb72a66d)


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
data = pd.read_csv('/content/drive/My Drive/Colab Notebooks/income(1) (1).csv',na_values=["?"])
data
![image](https://github.com/user-attachments/assets/eccb110b-c6a2-478b-b5da-e08524d77e94)


data.isnull().sum()

![image](https://github.com/user-attachments/assets/80b65fdf-47cd-4208-a231-5b452e1629f5)






missing=data[data.isnull().any(axis=1)]
missing

![image](https://github.com/user-attachments/assets/f1e734bf-b90d-47f7-a0ec-db7052d81771)


data2 = data.dropna(axis=0)
data2
![image](https://github.com/user-attachments/assets/7a7a4695-5b56-4b6a-82d8-b836907fa111)



sal=data['SalStat']
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])

![image](https://github.com/user-attachments/assets/53f6287e-cd08-419a-a611-5b1706838972)



sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs




data2
new_data=pd.get_dummies(data2, drop_first=True)
new_data
columns_list=list(new_data.columns)
print(columns_list)
![image](https://github.com/user-attachments/assets/1fd94e50-b911-4a50-b879-62100934a49f)



features=list(set(columns_list)-set(['SalStat']))
features=list(set(columns_list)-set(['SalStat']))


y=new_data['SalStat'].values
print(y)/

![image](https://github.com/user-attachments/assets/a759ada7-ea22-4fd9-b141-1c21307bfc36)


x = new_data[features].values
print(x)
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3, random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
prediction = KNN_classifier.predict(test_x)
confusionMmatrix = confusion_matrix(test_y, prediction)
print(confusionMmatrix)

![image](https://github.com/user-attachments/assets/2f22702c-1195-48bc-824f-709c47d8d007)



accuracy_score=accuracy_score(test_y, prediction)
print(accuracy_score)
misclassified Sample: 1540

print('Misclassified samples: %d' % (test_y != prediction).sum())


data.shape

(31978 ,13)
FEATURE SELECTION TECHNIQUES

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()

![image](https://github.com/user-attachments/assets/80c6f20d-bd11-46f2-9a15-1a9ab29cf2a9)



contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)

![image](https://github.com/user-attachments/assets/2e638f35-bbd7-4c97-a542-622f2d383cca)



chi2, p, _, _ = chi2_contingency(contingency_table)
print(f"Chi-Square Statistic: {chi2}")
print(f"P-value: {p}")

![image](https://github.com/user-attachments/assets/1b786fa6-b7f5-42bd-9285-76f6c973977a)




import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1':[1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3':[0,1,1,0,1],
    'Target' :[0,1,1,0,1]
}


df=pd.DataFrame(data)
X=df[['Feature1','Feature3']]
y=df['Target']

selector=SelectKBest(score_func=mutual_info_classif, k=1)
X_new = selector.fit_transform (X,y)


selected_feature_indices = selector.get_support(indices=True)
selected_features = X.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)

![image](https://github.com/user-attachments/assets/b0de4e57-9a59-4e9f-aaf3-408df241a0e2)











# RESULT:
      Feature selection and Feature Scaling of the given is done Successfully
       
