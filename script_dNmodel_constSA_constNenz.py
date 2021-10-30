from wild_type_model import WildType
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from constants import HRS_TO_SECS, OD_TO_COUNT_CONC

GC_ODs_N = pd.read_csv("data/GC_ODs_N.csv")
Time = GC_ODs_N.loc[:,'Time'].astype(np.float64)

# log transform and fit
WT_a_log10 = np.log10(GC_ODs_N.loc[:, 'dN_a'])

# Taken from https://stackoverflow.com/questions/55725139/fit-sigmoid-function-s-shape-curve-to-data-using-python
def sigmoid(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0)))+b
    return y

p0 = [max(WT_a_log10), np.median(Time), 1, min(WT_a_log10)]  # this is an mandatory initial guess
popt, pcov = curve_fit(sigmoid, Time, WT_a_log10, p0, method='dogbox')

fit_fun_log10 = lambda t: sigmoid(t, *popt)

# plot log10 data and spline
t = np.linspace(0, Time.iloc[-1] + 100, num=int(1e3))
plt.scatter(Time, WT_a_log10)
plt.plot(t, fit_fun_log10(t))
plt.legend(['data', 'Sigmoid'], loc='upper right')
plt.title('log(OD) fit to sigmoid function')
plt.show()

# plot untransformed data spline
fit_fun = lambda t: 10**fit_fun_log10(t)
plt.scatter(Time, np.power(10,WT_a_log10))
plt.plot(t, fit_fun(t))
plt.title('log(OD) fit to sigmoid function transformed')
plt.legend(['data', 'Sigmoid'], loc='upper right')
plt.show()

# create model: cylindrical MCP
# cell geometry
cell_radius = 0.375e-6
cell_length = 2.47e-6
cell_surface_area = 2*np.pi*cell_radius*cell_length
cell_volume = 4*np.pi/3*(cell_radius)**3 + np.pi*(cell_length - 2*cell_radius)*(cell_radius**2)

# WT MCP geometry
rmcp_eff = 7e-8     # [=] m
vmcp_eff = (4/3)*np.pi*(rmcp_eff**3)    # [=] m^3
nmcp_eff = 15 # [=] MCPs per cell

# MT geometry
radius_mcp = 2.5e-8 # [=] m
mcp_surface_area = 2*np.pi*(radius_mcp)*cell_length
mcp_volume = np.pi*(radius_mcp**2)*cell_length
# Conserved enzyme volume
#nmcp = nmcp_eff * (4/3) * rmcp_eff**3 / (cell_length * radius_mcp**2)
# TEM based
#nmcp = 5
# Conserved surface area
nmcp = nmcp_eff * 2 * rmcp_eff**2 / (cell_length * radius_mcp)

# external volume geometry
external_volume = 5e-5  # [=] m^3
wild_type_model = WildType(fit_fun, Time.iloc[-1], mcp_surface_area, mcp_volume,
                           cell_surface_area, cell_volume, external_volume)

PermMCPPolar = 10 ** -7.4     # [=] m/s
PermMCPNonPolar = 10 ** -7.4  # [=] m/s

# calculate Vmax parameters
# assume that the number of enzymes in the elongated PduMTs is same as MCPs,
# but in a different total volume
vmcp_eff = mcp_volume
NAvogadro = 6.02e23

# MCP || PduCDE || forward
kcatCDE = 300.   # [=] 1/s
N_CDE_MCP = 400.     # [=] enzymes per compartment, MCP case
CDE_tot = N_CDE_MCP * nmcp_eff  # [=] total number of enzymes in the cell, MCP case
N_CDE = CDE_tot / nmcp          # [=] number of CDE enzymes per MT
CDE_con = N_CDE / (NAvogadro * vmcp_eff)   # [=] mM
VmaxCDEf = kcatCDE * CDE_con # [=] mM/s

# MCP || PduP || forward
kcatPf = 55.        # [=] 1/s
N_P_MCP = 3*200.    # [=] enzymes per compartment, MCP case
P_tot = N_P_MCP * nmcp_eff  # [=] total number of P enzymes in cell, MCP case
N_P = P_tot / nmcp          # [=] numer of P enzymes per MT
P_con = N_P / (NAvogadro * vmcp_eff)    # [=] mM
VmaxPf = kcatPf * P_con     # [=] mM/s

# MCP || PduP || reverse
kcatPr = 6.     # [=] 1/s
VmaxPr = kcatPr * P_con     # [=] mM/s

# MCP || PduQ || forward
kcatQf = 55.    # [=] 1/s
N_Q_MCP = 3*150.        # [=] enzymes per compartment, MCP case
Q_tot = N_Q_MCP * nmcp_eff  # [=] total number of q enzymes in cell, MCP case
N_Q = Q_tot / nmcp          # [=] number of q enzymes per MT
Q_con = N_Q / (NAvogadro * vmcp_eff)    # [=] mM
VmaxQf = kcatQf * Q_con     # [=] mM/s

# MCP || PduQ || reverse
kcatQr = 6.     # [=] 1/s
VmaxQr = kcatQr * Q_con     # [=] mM/s

# cytosol || PduL || forward
kcatL = 100.    # [=] 1/s
L_con = 0.1     # [=] mM (ref: paper from Andre)
VmaxLf = kcatL * L_con      # [=] mM/s

# initialize parameters
params = {'PermMCPPropanediol': PermMCPPolar,
            'PermMCPPropionaldehyde': PermMCPNonPolar,
            'PermMCPPropanol': PermMCPPolar,
            'PermMCPPropionyl': PermMCPPolar,
            'PermMCPPropionate': PermMCPPolar,
            'nmcps': nmcp,
            'PermCellPropanediol': 10**-4,
            'PermCellPropionaldehyde': 10**-2,
            'PermCellPropanol': 10**-4,
            'PermCellPropionyl': 10**-5,
            'PermCellPropionate': 10**-7,
            'VmaxCDEf': VmaxCDEf,
            'KmCDEPropanediol': 0.5,
            'VmaxPf': VmaxPf,
            'KmPfPropionaldehyde': 15,
            'VmaxPr': VmaxPr,
            'KmPrPropionyl':  95,
            'VmaxQf': VmaxQf,
            'KmQfPropionaldehyde':  15,
            'VmaxQr': VmaxQr,
            'KmQrPropanol':  95,
            'VmaxLf': VmaxLf,
            'KmLPropionyl': 20}

# initialize initial conditions
init_conds = {'PROPANEDIOL_MCP_INIT': 0,
              'PROPIONALDEHYDE_MCP_INIT': 0,
              'PROPANOL_MCP_INIT': 0,
              'PROPIONYL_MCP_INIT': 0,
              'PROPIONATE_MCP_INIT': 0,
              'PROPANEDIOL_CYTO_INIT': 0,
              'PROPIONALDEHYDE_CYTO_INIT': 0,
              'PROPANOL_CYTO_INIT': 0,
              'PROPIONYL_CYTO_INIT': 0,
              'PROPIONATE_CYTO_INIT': 0,
              'PROPANEDIOL_EXT_INIT': 55,
              'PROPIONALDEHYDE_EXT_INIT': 0,
              'PROPANOL_EXT_INIT': 0,
              'PROPIONYL_EXT_INIT': 0,
              'PROPIONATE_EXT_INIT': 0}

# run model for parameter set
time_concat, sol_concat = wild_type_model.generate_time_series(init_conds, params)

# plot MCP solutions
plt.figure(3)
yext = sol_concat[:, :5]
plt.plot(time_concat/HRS_TO_SECS, yext)
plt.legend(['Propanediol', 'Propionaldehyde', 'Propanol', 'Propionyl', 'Propionate'], loc='upper right')
plt.title('Plot of MCP concentrations')
plt.xlabel('time (hr)')
plt.ylabel('concentration (mM)')
plt.show()

# plot cellular solution
plt.figure(4)
yext = sol_concat[:, 5:10]
plt.plot(time_concat/HRS_TO_SECS, yext)
plt.legend(['Propanediol', 'Propionaldehyde', 'Propanol', 'Propionyl', 'Propionate'], loc='upper right')
plt.title('Plot of cytosol concentrations')
plt.xlabel('time (hr)')
plt.ylabel('concentration (mM)')
plt.show()

# plot external solution
plt.figure(5)
yext = sol_concat[:, 10:]
plt.plot(time_concat/HRS_TO_SECS, yext)
plt.legend(['Propanediol', 'Propionaldehyde', 'Propanol', 'Propionyl', 'Propionate'], loc='upper right')
plt.title('Plot of external concentrations')
plt.xlabel('time (hr)')
plt.ylabel('concentration (mM)')
plt.show()

init_conds_list = np.array([val for val in init_conds.values()])

# conservation of mass formula
mcp_masses_org = init_conds_list[:5] * mcp_volume * params["nmcps"] * wild_type_model.optical_density_ts(Time.iloc[-1])\
                 * OD_TO_COUNT_CONC * external_volume
cell_masses_org = init_conds_list[5:10] * cell_volume * wild_type_model.optical_density_ts(Time.iloc[-1])* OD_TO_COUNT_CONC\
                  * external_volume
ext_masses_org = init_conds_list[10:] * external_volume

mcp_masses_fin = sol_concat[-1,:5] * mcp_volume * params["nmcps"] *wild_type_model.optical_density_ts(Time.iloc[-1]) \
                 * OD_TO_COUNT_CONC * external_volume
cell_masses_fin = sol_concat[-1,5:10] * cell_volume * wild_type_model.optical_density_ts(Time.iloc[-1]) * OD_TO_COUNT_CONC \
                  * external_volume
ext_masses_fin = sol_concat[-1,10:] * external_volume


print("Original mass: " + str(ext_masses_org.sum() + cell_masses_org.sum() + mcp_masses_org.sum()))
print("Final mass: " + str(ext_masses_fin.sum() + cell_masses_fin.sum() + mcp_masses_fin.sum()))

# save trajectories to a csv
#np.savetxt("dN_constSA_constNenz_t.csv",time_concat,delimiter=",")
#np.savetxt("dN_constSA_constNenz.csv",sol_concat,delimiter=",")