import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from impedance.visualization import plot_nyquist
from impedance.models.circuits import CustomCircuit


# Import your data.csv 
file_path = r'C:\Users\20224751\Downloads\EIS_data_py.csv'

df = pd.read_csv(file_path)
print(df.to_string()) 


# Remove white spaces from each column name
df.columns = [col.strip() for col in df.columns]


data = df.loc[(df['Im(Z)/Ohm'])>-0.06]


z=data['Re(Z)/Ohm'].values-1j*data['Im(Z)/Ohm'].values

f_data = df.loc[(df['freq/Hz'])<13100.0000] #<13000
a_f= np.array(f_data['freq/Hz'])
#print(z.shape)
#print(a_data.shape)


# Plot the data
fig, ax = plt.subplots()
ax.plot(z.real, -z.imag, marker='o',mfc='none',ls='none')
ax.set_xlabel(r'Z$_{real}$', size=14)
ax.set_ylabel(r'-Z$_{imag}$',size=14)
ax.set_aspect('equal')
plt.grid()


# Circuit for the fit
circuit = 'R0-p(R1,CPE1)-p(R2-Wo1,CPE2)'
initial_guess = [1e-9, .05, 1e-5, 0.1, .5, .05, 1, 1e-2, 0.8]

circuit = CustomCircuit(circuit, initial_guess=initial_guess)


# Fit results
circuit.fit(a_f, z)
print(circuit)

Z_fit = circuit.predict(a_f)
print(Z_fit)


Zreal_fit=Z_fit.real
Zimm_fit=-Z_fit.imag


zreal=z.real
zimag=-z.imag


# Plot the result of the fit
fig, ax = plt.subplots()
plt.scatter(zreal, zimag, label='Data')
plt.plot(Zreal_fit, Zimm_fit, label='Fitted data', color='red')
ax.set_xlabel(r'Z$_{real}$', size=14)
ax.set_ylabel(r'-Z$_{imag}$',size=14)
plt.title('Zreal vs -Zimm')
ax.set_aspect('equal')
plt.legend()
plt.grid()
plt.show()

