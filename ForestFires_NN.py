#!/usr/bin/env python
# coding: utf-8

# In[83]:


import pandas as pd
data=pd.read_csv("forestfires.csv")
data.head()


# In[84]:


data.columns


# In[85]:


data.shape


# In[86]:


data.info()


# In[87]:


data.isnull().any()


# In[88]:


data.isnull().any().sum()


# In[89]:


data['size_category'].unique()


# In[90]:


from keras.models import Sequential
from keras.layers import Dense, Activation,Layer,Lambda,Conv2D


# In[91]:


data.loc[data.size_category=="small",'size_category']=0


# In[92]:


data.loc[data.size_category=='large','size_category']=1


# In[93]:


data['size_category']


# In[94]:


data.head()


# In[95]:


data['size_category'].unique()


# In[96]:


data['size_category'].isnull().any()


# In[97]:


data['size_category'].value_counts().plot(kind='bar')


# In[98]:


data.drop("month",inplace=True, axis=1)


# In[99]:


data.drop("day",inplace=True,axis=1)


# In[100]:


data.head()


# In[101]:


data.shape


# In[102]:


data.info()


# In[103]:


data.head()


# In[104]:


from sklearn.model_selection import train_test_split
train,test=train_test_split(data,test_size=0.3, random_state=42)


# In[105]:


train,val=train_test_split(train,test_size=0.2)


# In[106]:


trainx=train.iloc[:,0:28]
trainy=train.iloc[:,28]
testx=test.iloc[:,0:28]
testy=test.iloc[:,28]
valx=val.iloc[:,0:28]
valy=val.iloc[:,28]


# In[107]:


model=Sequential()
model.add(Dense(32, activation='relu', input_shape=(28,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# In[108]:


model.compile(optimizer='sgd',loss='binary_crossentropy',metrics=['accuracy'])


# In[109]:


hist=model.fit(trainx,trainy,batch_size=32, epochs=100,validation_data=(valx,valy))


# In[110]:


#To get the accuracy we use indexing[1] as model.evaluate returns loss([0]) and accuracy([1])
#Testing accuaracy
model.evaluate(testx,testy)[1]


# In[111]:


#Training accuracy
model.evaluate(trainx,trainy)[1]


# In[112]:


#Visualizing the model
import matplotlib.pyplot as plt
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()


# In[113]:


import pydot
from keras.utils import plot_model
plot_model(model,to_file="first_model.png")


# # After minmaxscaler

# In[114]:


from sklearn.preprocessing import MinMaxScaler
scale=MinMaxScaler()
trainx=scale.fit_transform(trainx)
testx=scale.fit_transform(testx)
valx=scale.fit_transform(valx)


# In[115]:


trainx


# In[116]:


testx


# In[117]:


valx


# In[118]:


model1=Sequential()
model1.add(Dense(32, activation='relu', input_shape=(28,)))
model1.add(Dense(32, activation='relu'))
model1.add(Dense(1, activation='sigmoid'))


# In[119]:


model1.compile(optimizer='sgd',loss='binary_crossentropy',metrics=['accuracy'])


# In[120]:


hist1=model1.fit(trainx,trainy,batch_size=32, epochs=100,validation_data=(valx,valy))


# In[121]:


#To get the accuracy we use indexing[1] as model.evaluate returns loss([0]) and accuracy([1])
#Testing accuaracy
model1.evaluate(testx,testy)[1]


# In[122]:


#Training accuracy
model1.evaluate(trainx,trainy)[1]


# In[123]:


#Visualizing the model
import matplotlib.pyplot as plt
plt.plot(hist1.history['accuracy'])
plt.plot(hist1.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()


# In[124]:


import pydot
from keras.utils import plot_model
plot_model(model1,to_file="second_model.png")


# In[ ]:




