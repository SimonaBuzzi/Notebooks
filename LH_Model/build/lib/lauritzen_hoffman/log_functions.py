import numpy as np 
import  matplotlib.pyplot as plt



def logDH(x_1, x_2, x_3, x_4, y_1, y_2, y_3, y_4):

    k_lin_1, n_lin_1 = np.polyfit(x_1,y_1,1)
    k_lin_2, n_lin_2 = np.polyfit(x_2,y_2,1)
    k_lin_3, n_lin_3 = np.polyfit(x_3,y_3,1)
    k_lin_4, n_lin_4 = np.polyfit(x_4,y_4,1)

    coef_1 = k_lin_1, n_lin_1
    coef_2 = k_lin_2, n_lin_2
    coef_3 = k_lin_3, n_lin_3
    coef_4 = k_lin_4, n_lin_4

    plt.figure()
    plt.plot(x_1, y_1, 'bs', label='t_1')
    plt.plot(x_2, y_2, 'ok', label='t_2')
    plt.plot(x_3, y_3, '<', label='t_3')
    plt.plot(x_4, y_4, 'v', label='t_4')
    plt.plot(x_1, np.poly1d(coef_1)(x_1), linewidth=1, color='red', label='Fit_1')
    plt.plot(x_2, np.poly1d(coef_2)(x_2), linewidth=1, color='red', label='Fit_2')
    plt.plot(x_3, np.poly1d(coef_3)(x_3), linewidth=1, color='red', label='Fit_3')
    plt.plot(x_4, np.poly1d(coef_4)(x_4), linewidth=1, color='red', label='Fit_4')
    plt.xlabel('t - tzero (min)')
    plt.ylabel('1 - (H/H0)')
    plt.title('Linear Fit')
    plt.legend()
    plt.grid(True)
    plt.show()


    return k_lin_1, n_lin_1, k_lin_2, n_lin_2, k_lin_3, n_lin_3, k_lin_4, n_lin_4



def logDH_fitting(df, tzero_1, tzero_2, tzero_3, tzero_4, DHinf_1, DHinf_2, DHinf_3, DHinf_4):
    
    t_1 = np.array(df['time_1']) #time conversion
    h_1 = np.array(df['Area_1'])
    t_2 = np.array(df['time_2'])
    h_2 = np.array(df['Area_2'])
    t_3 = np.array(df['time_3'])
    h_3 = np.array(df['Area_3'])
    t_4 = np.array(df['time_4'])
    h_4 = np.array(df['Area_4'])

    x_1 = np.abs(t_1 - tzero_1)
    x_2 = np.abs(t_2 - tzero_2)
    x_3 = np.abs(t_3 - tzero_3)
    x_4 = np.abs(t_4 - tzero_4)

    new_h_1 = h_1[h_1 < DHinf_1]
    new_h_2 = h_2[h_2 < DHinf_2]
    new_h_3 = h_3[h_3 < DHinf_3]
    new_h_4 = h_4[h_4 < DHinf_4]

    x_1 = x_1[h_1 < DHinf_1]
    x_2 = x_2[h_2 < DHinf_2]
    x_3 = x_3[h_3 < DHinf_3]
    x_4 = x_4[h_4 < DHinf_4]

    y_1 = np.log(1 - new_h_1/ DHinf_1)
    y_2 = np.log(1 - new_h_2/ DHinf_2)
    y_3 = np.log(1 - new_h_3/ DHinf_3)
    y_4 = np.log(1 - new_h_4/ DHinf_4)

    return x_1, x_2, x_3, x_4, y_1, y_2, y_3, y_4

   
   
def kg_fit(T_range, k_lin_1, k_lin_2, k_lin_3, k_lin_4, area, U, R, Tinf, T0m):

    T = np.array(T_range)
    T = T + 273.15
    slope = np.array((k_lin_1, k_lin_2, k_lin_3, k_lin_4))
    logI = np.abs(slope/area)
    log_I = np.log(logI)
    Unorm = U/(R*(T-Tinf))
    y = log_I + Unorm
    f = 2*T/(T+T0m)
    x = 1/(T*((T0m-T)*f)**2)  
    coef = np.polyfit(x,y,1)
    kg, lgI0 = coef
    plt.figure()
    plt.plot(x, y, 'bs', label='Data')
    plt.plot(x, np.poly1d(coef)(x), linewidth=2, label='Fit')
    plt.xlabel('1/T(DTf)^2')
    plt.ylabel('lg(I) + U/R(T-Tinf)')
    plt.title('Linear Fit for kg and lgI0')
    plt.legend()
    plt.grid(True)
    plt.show()
    return  kg, lgI0

