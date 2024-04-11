from lauritzen_hoffman.linear_rate import lh_growth_model

# file_path
file_path = r'C:\Users\20224751\Documents\Excel_files\PCL_data.csv'
initial_guess = 6, 0.05, 1, 1 
#parameters
Area = 1.300149 * (10 ** (-12))
T_range = [40, 42, 44, 46]
U = 1500
Tinf = 183.15
T0m = 355.15
R = 1.99

kg, lgI0 = lh_growth_model(file_path, initial_guess, T_range, Area, U, R, Tinf, T0m)