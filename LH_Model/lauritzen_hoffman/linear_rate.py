from lauritzen_hoffman.load import process_data
from lauritzen_hoffman.functions import DH_fitting
from lauritzen_hoffman.log_functions import logDH, logDH_fitting, kg_fit




def lh_growth_model(file_path, initial_guess, T_range, Area, U, R, Tinf, T0m):

    df = process_data(file_path)

    DHinf_1, k_1, tzero_1, n_1, covariance_1, DHinf_2, k_2, tzero_2, n_2, covariance_2, DHinf_3, k_3, tzero_3, n_3, covariance_3, DHinf_4, k_4, tzero_4, n_4, covariance_4 = DH_fitting(df, initial_guess)
    print("DHinf_1, k_1, tzero_1, n_1:", DHinf_1, k_1, tzero_1, n_1, "covariance_1:", covariance_1)
    print("DHinf_2, k_2, tzero_2, n_2:", DHinf_2, k_2, tzero_2, n_2, "covariance_2:", covariance_2)
    print("DHinf_3, k_3, tzero_3, n_3:", DHinf_3, k_3, tzero_3, n_3, "covariance_3:", covariance_3)
    print("DHinf_4, k_4, tzero_4, n_4:", DHinf_4, k_4, tzero_4, n_4, "covariance_4:", covariance_4)

    
    x_1, x_2, x_3, x_4, y_1, y_2, y_3, y_4 = logDH_fitting(df, tzero_1, tzero_2, tzero_3, tzero_4, DHinf_1, DHinf_2, DHinf_3, DHinf_4)
    k_lin_1, n_lin_1, k_lin_2, n_lin_2, k_lin_3, n_lin_3, k_lin_4, n_lin_4 = logDH(x_1, x_2, x_3, x_4, y_1, y_2, y_3, y_4)
    print("slope_1:", k_lin_1, "n_calculated_1:", n_lin_1)
    print("slope_2:", k_lin_2, "n_calculated_2:", n_lin_2)
    print("slope_3:", k_lin_3, "n_calculated_3:", n_lin_3)
    print("slope_4:", k_lin_4, "n_calculated_4:", n_lin_4)


    kg, lgI0 = kg_fit(T_range, k_lin_1, k_lin_2, k_lin_3, k_lin_4, Area, U, R, Tinf, T0m)
    print ("kg:", kg, "lgI0:", lgI0 )

    return kg, lgI0 

