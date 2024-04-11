# Usage:
# python -m rob_copy C:\Users\20224751\Downloads\SN_TEST.csv 6.19 0.05 1 1 

import argparse

from nucleation.enthalpy_fit import fitted_enthalpy
from nucleation.plot import scatter_plot
from nucleation.plot import plot_results
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit


parser = argparse.ArgumentParser(
                    prog='Enthalpy_fitter',
                    description='For all your enthalpy fitting needs',
                    epilog='Thanks for using Enthalpy Fitter!!!')

parser.add_argument('filename')  
parser.add_argument('DHinf') 
parser.add_argument('k') 
parser.add_argument('tzero') 
parser.add_argument('n') 
parser.add_argument('-l', '--log',
                    action='store_true')  

args = parser.parse_args()


def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """Load and clean the data
    """
    df = pd.read_csv(file_path)
    df.columns = [col.strip() for col in df.columns]
    return df


def nucleation_fit(file_path, initial_guess):
    
    df = load_and_clean_data(file_path)
    scatter_plot(df)
   
    DHinf_fit, k_fit, tzero_fit, n_fit = fitted_enthalpy(df, initial_guess) 
    fit_plot(df, DHinf_fit, k_fit, tzero_fit, n_fit)
    


    

if __name__ == '__main__':
     
     file_path:  r'C:\Users\20224751\Downloads\SN_TEST.csv '
     initial_guess: 
     nucleation_fit(file_path, initial_guess)
 
   
