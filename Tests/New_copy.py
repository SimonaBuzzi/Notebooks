#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

file_path = r'C:\Users\20224751\Downloads\SN_TEST.csv'

df = pd.read_csv(file_path)

print(df.to_string()) 


# In[2]:


df = pd.DataFrame(df)
df.columns = [col.strip() for col in df.columns]
print(df.columns)
print(df)


# In[3]:


df.plot(x='time', y='Area', marker='o', linestyle='-')
plt.xlabel('Time (min)')
plt.ylabel('DH (J/g)')
plt.title('Enthalpy vs Time')
plt.show()


# In[4]:


t_data = np.array(df['time'])
a_data = np.array(df['Area'])


# In[5]:


# Define the DH function
def DH(t_data, DHinf, k, tzero, n):
    return DHinf * (1 - np.exp(-k * np.abs(t_data - tzero)**n))
# ou need to provide an initial guess for the parameters (DHinf, k, tzero, and n). This is necessary for the curve fitting algorithm to start the optimization process. You might base your initial guess on some knowledge of the system or by visually inspecting the data.
initial_guess = (6.19, 0.05, 1, 1)


# In[7]:


# Fit the function 
from scipy.optimize import curve_fit

#scipy.optimize.curve_fit(f, xdata, ydata, p0=None)
params, covariance = curve_fit(DH, t_data, a_data, p0=initial_guess)

# Fitted parameters estraction 
DHinf_fit, k_fit, tzero_fit, n_fit = params


# In[8]:


print("Fitted DHinf:", DHinf_fit)
print("Fitted k:", k_fit)
print("Fitted tzero:", tzero_fit)
print("Fitted n:", n_fit)


# In[8]:


print(covariance)

def fitted_DH(t_data):
    return DH(t_data, DHinf_fit, k_fit, tzero_fit, n_fit)

plt.scatter(t_data, a_data, label='Data')
plt.plot(t_data, fitted_DH(t_data), label='Fitted DH Function', color='red')
plt.xlabel('Time (min)')
plt.ylabel('DH (J/g)')
plt.title('Enthalpy vs Time Fit')
plt.legend()
plt.show()

# In[9]:

x = np.abs(t_data - tzero_fit)
y = np.log(1 - a_data / DHinf_fit)

def LogDHn(x, k_lin, n_lin):
    return -k_lin * np.abs(x**n_lin)

coef = np.polyfit(x,y,1)
poly1d_fn = np.poly1d(coef)
print(coef)

# In[10]:

plt.plot(x, y, 'ro', label='Data')
plt.plot(x, poly1d_fn(x), '--k', label='Fit')
plt.legend()
plt.xlabel('t-tzero')
plt.ylabel('1-ln(H1/H0)')
plt.title('Linear fit')



# In[38]:



# In[40]:




# In[41]:



# In[ ]:




