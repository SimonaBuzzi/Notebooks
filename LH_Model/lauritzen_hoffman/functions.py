import numpy as np
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit 



# Enthalpy base function
def DH(t_data, DHinf, k, tzero, n):
    return DHinf * (1 - np.exp(-k * np.abs(t_data - tzero)**n))

def fitted_DH(t_data, DHinf_fit, k_fit, tzero_fit, n_fit):
    return DH(t_data, DHinf_fit, k_fit, tzero_fit, n_fit)

# DH_fitting
def DH_fitting(df, initial_guess):
    
    t_1 = np.array(df['time_1']) #time conversion
    h_1 = np.array(df['Area_1'])
    t_2 = np.array(df['time_2'])
    h_2 = np.array(df['Area_2'])
    t_3 = np.array(df['time_3'])
    h_3 = np.array(df['Area_3'])
    t_4 = np.array(df['time_4'])
    h_4 = np.array(df['Area_4'])
    
    params_1, covariance_1 = curve_fit(DH, t_1, h_1, p0=initial_guess) #scipy curve fitting
    params_2, covariance_2 = curve_fit(DH, t_2, h_2, p0=initial_guess)
    params_3, covariance_3 = curve_fit(DH, t_3, h_3, p0=initial_guess)
    params_4, covariance_4 = curve_fit(DH, t_4, h_4, p0=initial_guess)
    
    DHinf_1, k_1, tzero_1, n_1 = params_1 #parameters 
    DHinf_2, k_2, tzero_2, n_2 = params_2
    DHinf_3, k_3, tzero_3, n_3 = params_3
    DHinf_4, k_4, tzero_4, n_4 = params_4

    h_1_fitted = fitted_DH(t_1, DHinf_1, k_1, tzero_1, n_1)
    h_2_fitted = fitted_DH(t_2, DHinf_2, k_2, tzero_2, n_2)
    h_3_fitted = fitted_DH(t_3, DHinf_3, k_3, tzero_3, n_3)
    h_4_fitted = fitted_DH(t_4, DHinf_4, k_4, tzero_4, n_4)

    plt.figure()
    plt.plot(t_1, h_1, 'bs', label='t_1')
    plt.plot(t_2, h_2, 'ok', label='t_2')
    plt.plot(t_3, h_3, '<', label='t_3')
    plt.plot(t_4, h_4, 'v', label='t_4')
    plt.plot(t_1, h_1_fitted, color='red', label='Fit_1')
    plt.plot(t_2, h_2_fitted, color='red', label='Fit_2')
    plt.plot(t_3, h_3_fitted, color='red',label='Fit_3')
    plt.plot(t_4, h_4_fitted, color='red', label='Fit_4')
    plt.xlabel('time(min)')
    plt.ylabel('Enthalpy (J/g)')
    plt.title('Scatter plot')
    plt.legend()
    plt.grid(True)
    plt.show()
    return  DHinf_1, k_1, tzero_1, n_1, covariance_1, DHinf_2, k_2, tzero_2, n_2, covariance_2, DHinf_3, k_3, tzero_3, n_3, covariance_3, DHinf_4, k_4, tzero_4, n_4, covariance_4


