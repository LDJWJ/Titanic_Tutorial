#!/usr/bin/env python
# coding: utf-8

# ### 나의 첫 모델 만들기
# 
# ### 학습 내용
# * 1-1 데이터 불러오기
# * 1-2 데이터 탐색하기
# * 1-3 모델 만들고 제출해 보기
# 
# ### 준비
# * (내 컴퓨터-로컬, 구글 콜랩의 경우) 캐글 데이터 셋을 다운로드(https://www.kaggle.com/c/titanic/data) 받는다.
# * 구글 콜랩의 경우, 데이터 셋을 올린다.
# 
# * (Kaggle Notebook의 경우) 경로를 확인 후, 데이터를 로드시에 해당 경로로 불러온다.
# 

# In[21]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

## 캐글 커널에서 실행 시, 파일의 경로 확인
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[4]:


# 내 컴퓨터에 데이터가 있을 경우,
# train = pd.read_csv("train.csv")
# test = pd.read_csv("test.csv")

# Kaggle Notebook 의 경우, 
train = pd.read_csv("/kaggle/input/titanic/train.csv")
test = pd.read_csv("/kaggle/input/titanic/test.csv")


# In[5]:


print(train.shape)
print(test.shape)


# In[6]:


print(train.columns)
print(test.columns)


# In[7]:


train.dtypes


# In[8]:


print(train.info())
print(test.info())


# In[9]:


train.isnull().sum()


# In[10]:


test.isnull().sum()


# In[11]:


train.describe()


# ### 1-3 모델 만들고 제출해 보기

# In[12]:


from sklearn.linear_model import LogisticRegression


# In[13]:


train.columns


# In[14]:


# 데이터 준비 - 빠른 모델 생성을 위해 처리 없이 가능한 변수만 선택
# 'Survived'를 제외 , 
# 'Embarked', 'Sex'',Name', 'Ticket' =>문자포함
#  'Age',
sel = ['PassengerId', 'Pclass', 'SibSp', 'Parch' ]

# 학습에 사용될 데이터 준비 X_train, y_train
X_train = train[sel]
y_train = train['Survived']
X_test = test[sel]


# In[15]:


model = LogisticRegression()


# In[16]:


# 학습
model.fit(X_train, y_train)


# ## 예측

# In[17]:


predictions = model.predict(X_test)
predictions[:15]


# In[18]:


# sub = pd.read_csv("gender_submission.csv")
sub = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
sub.head(15)


# In[19]:


sub['Survived'] = predictions
sub.head(15)


# In[20]:


sub.to_csv("logistic_first_model.csv", index=False)


# ## 제출방법
# ### 01. 로컬에서 실행 후, 제출시
# * https://www.kaggle.com/c/titanic 에 접속
# * 'Submit Predictions'을 선택 후, 해당 제출란에 생성된 *.csv 파일을 제출(드래그앤 드롭 또는 업로드 버튼을 이용)
# * 해당 제출 답에 대한 정확도를 확인
# 
# ### 02. Kaggle Notebook 에서 실행 후, 제출시
# * 코드 실행 'Commit'를 누르고, 다 실행이 마치면,
# * 'Open Version'을 선택 후, Output 메뉴 선택 후, 파일 선택 후, 제출

# In[ ]:




