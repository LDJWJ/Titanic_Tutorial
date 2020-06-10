#!/usr/bin/env python
# coding: utf-8

# ## 02. 데이터 전처리 및 EDA - AGE, Sex, Embarked 결측치(비어 있는 값) 처리

# ### 학습 내용
#  * 데이터 확인 및 전처리

# ### 나이와 승선항을 결측치 처리 후, 확인해 보자.

# In[71]:


## 설치가 안되어 있을 경우, 설치 
get_ipython().system('pip install missingno')


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno   # No module named 'missingno' 발생시, 위의 pip install missingno 설치 필요


# ## 01. EDA(탐색적 데이터 탐색)
#  * 데이터에 익숙해 지기
#  * 데이터 자료형에 대해 알아가기
#  * 데이터 컬럼명 알아보기

# In[13]:


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


# In[20]:


print(train.shape, test.shape)    # 데이터의 행과열


# In[14]:


## 데이터 확인
train.head()


# In[15]:


# 만약 전체 열이 확인 안 될 때,
for col in train.columns:
    print("column : ", col)
    print(train[col].head()) 
    print()


# ### 데이터 요약

# In[19]:


train.describe()


# ### 데이터 결측치 확인

# In[17]:


train.info()


# ### 결측치 확인
#  * figsize로 크기 설정
#  * seaborn의 heatmap 이용 결측치 확인 (cbar : colorbar, cmap : 색 지정, yticklabels : y축 유무)

# In[74]:


plt.figure(figsize=(10,7))
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap="GnBu")  # cbar : colorbar를 그리지 않음.


# In[75]:


plt.figure(figsize=(10,7))
sns.heatmap(test.isnull(), yticklabels=False, cbar=False, cmap="summer")  # cbar : colorbar를 그리지 않음.


# ### [더 알아보기] 데이터의 수치형 변수, 범주형 변수 살펴보기

# In[21]:


len(train.columns)


# ### 수치형 변수 살펴보기

# In[22]:


num_cols = [col for col in train.columns[:12] if train[col].dtype in ['int64', 'float64'] ]
train[num_cols].describe()


# ### 범주형 변수 살펴보기

# In[23]:


cat_cols = [col for col in train.columns[:12] if train[col].dtype in ['O'] ]
train[cat_cols].describe()


# ### 생존자 사망자의 비율이 얼마나 될까?

# In[24]:


sns.set_style('whitegrid')   # seaborn 스타일 지정
sns.countplot(x='Survived', data=train)


# In[25]:


## 해보기 : PClass 별 Count
sns.countplot(x='Pclass', data=train)


# ### 나이에 대해 살펴보자

# In[26]:


sns.distplot(train['Age'].dropna(), bins=30).set_xlim(0,)


# In[31]:


## 해보기 Fare
sns.distplot(test['Fare'].dropna(), bins=30).set_xlim(0,)


# * plt.subplots(행, 열, figsize=(크기지정))

# In[32]:


f,ax=plt.subplots(1,2,figsize=(18,8))

# 첫번째 그래프
sns.distplot(train['Age'].dropna(), bins=30, ax=ax[0])
ax[0].set_title('train - Age')

# 두번째 그래프 
sns.distplot(train['Age'].dropna(), bins=30, ax=ax[1])
ax[1].set_title('test - Age')
plt.show()


# ### 결측치 처리 첫번째
#  * 나이는 평균값으로 처리하자.
#  * 결측치 값을 채우기 -  usage : data['열이름'].fillna(값)

# In[33]:


train['Age'] = train['Age'].fillna(train['Age'].mean())
test['Age'] = test['Age'].fillna(test['Age'].mean())


# In[34]:


## 해보기 
test['Fare'] = test['Fare'].fillna(test['Fare'].mean())


# In[35]:


print(train.isnull().sum())
print(test.isnull().sum())


# ### 결측치 처리 두번째 Embarked(승선항)
#  * 가장 많이 나온 값으로 결측치 처리를 하자
#  * 범주(구분,종류)별 데이터 개수 => [Syntax] 데이터셋명['컬럼명'].value_counts() 

# In[84]:


val_Embarked = train['Embarked'].value_counts() 
val_Embarked


# In[85]:


val_Embarked.index[0]   #  행 이름 첫번째


# In[86]:


train['Embarked'] = train['Embarked'].fillna('S')


# In[87]:


print(train.isnull().sum())
print(test.isnull().sum())


# ### 데이터 전처리

# In[88]:


train.info()


# In[89]:


print( train['Sex'].value_counts() )
print( train['Embarked'].value_counts() )


#  * 데이터 자료형 변환
#  * 데이터.astype(변환될 자료형명)

# In[90]:


train['Sex'] = train['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
test['Sex'] = test['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

train['Embarked'] = train['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
test['Embarked']= test['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)


# In[91]:


## 나이에 대한 int 처리
train['Age'] = train['Age'].astype('int')
test['Age'] = test['Age'].astype('int')


# In[92]:


print(train.columns)
print(train.info())


# In[93]:


# 'Name', 'Ticket' =>  문자포함
sel = ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'SibSp','Parch', 'Embarked' ]

# 학습에 사용될 데이터 준비 X_train, y_train
X_train = train[sel]
y_train = train['Survived']
X_test = test[sel]


# ## 컬럼과 컬럼 사이의 관계 확인(상관계수 Heatmap)

# In[42]:


colormap = plt.cm.RdBu
plt.figure(figsize=(14, 12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train.corr(), annot=True, annot_kws={"size": 13})


# In[96]:


#correlation heatmap of dataset
def correlation_heatmap(df):
    _ , ax = plt.subplots(figsize =(14, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    
    _ = sns.heatmap(
        df.corr(), 
        cmap = colormap,
        square=True, 
        cbar_kws={'shrink':.9 }, 
        ax=ax,
        annot=True, 
        linewidths=0.1,vmax=1.0, linecolor='white',
        annot_kws={'fontsize':12 }
    )
    
    plt.title('Pearson Correlation of Features', y=1.05, size=15)

correlation_heatmap(train)


# ## 2-3 의사결정 트리 모델 만들고 제출해 보기
#  * 모델을 생성 후, 학습
#  * 그리고 예측을 수행 후, 제출한다.

# In[98]:


print(X_train.columns)
print(X_test.columns)


# In[102]:


from sklearn.tree import DecisionTreeClassifier
decisiontree = DecisionTreeClassifier()
decisiontree.fit(X_train, y_train)


# In[103]:


# 예측
predictions = decisiontree.predict(X_test)
predictions[:15]


# In[104]:


test_passengerId = test['PassengerId']
pred = predictions.astype(int)
df_pred = pd.DataFrame({'PassengerID':test_passengerId, 'Survived':pred})
df_pred.to_csv("decision_first_model.csv", index=False)


# ## 2-3 의사결정 트리 모델 - 'Fare'변수 추가
#  * 모델을 생성 후, 학습
#  * 그리고 예측을 수행 후, 제출한다.

# In[117]:


# 'Name', 'Ticket' =>  문자포함
sel = ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'SibSp','Parch', 'Embarked', 'Fare' ]

# 학습에 사용될 데이터 준비 X_train, y_train
X_train = train[sel]
y_train = train['Survived']
X_test = test[sel]


# In[118]:


from sklearn.tree import DecisionTreeClassifier
decisiontree = DecisionTreeClassifier()
decisiontree.fit(X_train, y_train)
# 예측
predictions = decisiontree.predict(X_test)
predictions[:15]


# In[119]:


test_passengerId = test['PassengerId']
pred = predictions.astype(int)
df_pred = pd.DataFrame({'PassengerID':test_passengerId, 'Survived':pred})
df_pred.to_csv("decision_second_model.csv", index=False)


# ## 직접 실습 해보기 : Logistic 회귀 모델 만들기

# In[120]:


from sklearn.linear_model import LogisticRegression


# In[121]:


model = LogisticRegression()
# 학습
model.fit(X_train, y_train)
# 예측
predictions = model.predict(X_test)
predictions[:15]


# In[122]:


test_passengerId = test['PassengerId']
pred = predictions.astype(int)
df_pred = pd.DataFrame({'PassengerID':test_passengerId, 'Survived':pred})
df_pred.to_csv("logistic_second_model.csv", index=False)  


# ### REF
# seaborn heatmap cmap : https://pod.hatenablog.com/entry/2018/09/20/212527 <br>
# seaborn set_style : https://www.codecademy.com/articles/seaborn-design-i

# In[ ]:




