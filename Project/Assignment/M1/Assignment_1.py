#!/usr/bin/env python
# coding: utf-8

# # Basic Python

# ## 1. Split this string

# In[5]:


s = "Hi there Sam!"
a=s.split(" ")
a


# In[ ]:





# In[ ]:





# In[ ]:





# *`italicized text`*## 2. Use .format() to print the following string. 
# 
# ### Output should be: The diameter of Earth is 12742 kilometers.

# In[7]:


planet = "Earth"
diameter = 12742
print("The diameter of {} is {} kilometers.".format(planet,diameter))


# In[ ]:





# ## 3. In this nest dictionary grab the word "hello"

# 

# In[8]:


d = {'k1':[1,2,3,{'tricky':['oh','man','inception',{'target':[1,2,3,'hello']}]}]}
print(d["k1"][3]["tricky"][3]["target"][3])


# # Numpy

# In[ ]:


import numpy as np


# ## 4.1 Create an array of 10 zeros? 
# ## 4.2 Create an array of 10 fives?

# In[10]:


import numpy as np
array=np.zeros(10)
print(array)


# In[12]:


import numpy as np
array=np.ones(10)*5
print(array)


# ## 5. Create an array of all the even integers from 20 to 35

# In[13]:


import numpy as np
array=np.arange(20,35,2)
print(array) 


# ## 6. Create a 3x3 matrix with values ranging from 0 to 8

# In[14]:


import numpy as np
x =  np.arange(0, 9).reshape(3,3)
print(x)


# ## 7. Concatinate a and b 
# ## a = np.array([1, 2, 3]), b = np.array([4, 5, 6])

# In[15]:


import numpy as np
a,b = np.array([1, 2, 3]),np.array([4, 5, 6])
g = np.concatenate((a, b), axis = 0)
print (g)


# # Pandas

# ## 8. Create a dataframe with 3 rows and 2 columns

# In[9]:


import pandas as pd
data = [['Loki', 100], ['Sugumar', 200], ['Bharathi', 140]]
df = pd.DataFrame(data, columns=['Name', 'Score'])
df


# In[10]:


import pandas as pd
data = [['Prem', 100], ['Sri', 200], ['Subam', 140]]
df = pd.DataFrame(data, columns=['Name', 'Score'])
df


# ## 9. Generate the series of dates from 1st Jan, 2023 to 10th Feb, 2023

# In[11]:


import pandas as pd
  
per1 = pd.date_range(start ='1-1-2023', 
         end ='10-02-2023')
  
for val in per1:
    print(val)


# ## 10. Create 2D list to DataFrame
# 
# lists = [[1, 'aaa', 22],
#          [2, 'bbb', 25],
#          [3, 'ccc', 24]]

# In[ ]:





# In[12]:


import pandas as pd  
lists = [[1, 'aaa', 22], [2, 'bbb', 25], [3, 'ccc', 24]]
df = pd.DataFrame(lists, columns =['Player', 'Name','Number']) 
print(df )


# In[ ]:




