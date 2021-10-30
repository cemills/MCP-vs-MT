'''
The wild-type model contains native reaction pathway
in the MCP; diffusion in the cell; diffusion from the cell 
in the external volume.

This model is currently in use. The DhaB-DhaT model assumes that there 
are M identical MCPs within the cytosol and N identical growing cells within the
external volume. From time scale analysis, gradients in the cell are removed.

Programme written by aarcher07
Editing History: See github history
'''


import numpy as np
from scipy.integrate import solve_ivp, quad
import scipy.constants as constants
import sympy as sp
import scipy.sparse as sparse
import time
import matplotlib.pyplot as plt
from constants import HRS_TO_SECS, VARIABLE_INIT_NAMES, OD_TO_COUNT_CONC, MODEL_PARAMETER_LIST

class WildType:
    def __init__(self, optical_density_ts, fin_exp_time, mcp_surface_area, mcp_volume,
                 cell_surface_area, cell_volume, external_volume):
        """
        Initializes parameters to be used numerical scheme
        :param optical_density_ts: optical density at any given time during experiment (in OD)
        :param fin_exp_time: duration of the experiment (in hours)
        :param mcp_surface_area: surface area of cell (in metres^2)
        :param mcp_volume: volume of microcompartment (in metres^3)
        :param cell_surface_area: cell surface area (in metres^2)
        :param cell_volume: cell volume (in metres^3)
        :param external_volume: external volume amount (in metres^3)
        """

        # geometric parameters
        self.external_volume = external_volume
        self.mcp_surface_area = mcp_surface_area
        self.mcp_volume = mcp_volume
        self.cell_surface_area = cell_surface_area
        self.cell_volume = cell_volume

        # geometric ratio
        self.MCP_surf_MCP_vol_ratio = mcp_surface_area / mcp_volume 
        self.cell_surf_cell_vol_ratio = cell_surface_area / cell_volume 
        self.MCP_surf_cell_vol_ratio = mcp_surface_area / cell_volume 
        self.cell_surf_external_vol_ratio = cell_surface_area / external_volume 

        # differential equation parameters
        self.nvars = 5*3
        self.optical_density_ts = optical_density_ts
        self.fin_exp_time = fin_exp_time

        # set jacobian for ODE integration
        self._set_symbolic_state_vars()
        self._set_param_sp_symbols()
        # self._set_symbolic_sderiv_conc_fun()

    def _set_symbolic_state_vars(self):
        """
        Generates the symbol state variables for the model
        """
        self.x_sp = np.array(sp.symbols('x:' + str(self.nvars)))

    def _set_param_sp_symbols(self):
        """
        sets dictionary of parameters to be analyzed using sensitivity analysis
        """
        self.params_sens_sp_dict = {name: sp.symbols(name) for name in MODEL_PARAMETER_LIST}
        self.params_sens_sp = list((self.params_sens_sp_dict).values())

    def _sderiv(self,t,x,params=None):
        """
        Computes the spatial derivative of the system at time point, t
        :param t: time
        :param x: state variables
        :param params: parameter list
        """

        if params is None:
            if self.params is None:
                print("Parameters have not been set.")
                return
            params = self.params

        ###################################################################################
        ################################# Initialization ##################################
        ###################################################################################
     

        # Integration Parameters
        assert len(x) == self.nvars
        n_compounds_cell = 5
        d = np.zeros((len(x))).tolist()  # convert to list to allow use of symbolic derivatives

        # differential equation parameters
        ncells = self.optical_density_ts(t/HRS_TO_SECS) * OD_TO_COUNT_CONC * self.external_volume
        nmcps = params['nmcps']
        

        ###################################################################################
        ################################## MCP reactions ##################################
        ###################################################################################
        R_CDE = params["VmaxCDEf"]*x[0]/(x[0] + params["KmCDEPropanediol"])
        R_Pf = params["VmaxPf"]*x[1]/(x[1] + params["KmPfPropionaldehyde"])
        R_Pr = params["VmaxPr"]*x[3]/(x[3] + params["KmPrPropionyl"])
        R_Qf = params["VmaxQf"]*x[1]/(x[1] + params["KmQfPropionaldehyde"])
        R_Qr = params["VmaxQr"]*x[2]/(x[2] + params["KmQrPropanol"])
        R_Lf = params["VmaxLf"]*x[8]/(x[8] + params["KmLPropionyl"])


        d[0] = -R_CDE + self.MCP_surf_MCP_vol_ratio *params['PermMCPPropanediol'] * (x[0 + n_compounds_cell] - x[0])  # microcompartment equation for G
        d[1] =  R_CDE -  R_Pf - R_Qf + R_Pr + R_Qr +self.MCP_surf_MCP_vol_ratio * params['PermMCPPropionaldehyde']* (x[1 + n_compounds_cell] - x[1])  # microcompartment equation for H
        d[2] = R_Qf - R_Qr + self.MCP_surf_MCP_vol_ratio * params['PermMCPPropanol'] * (x[2 + n_compounds_cell] - x[2])  # microcompartment equation for P
        d[3] = R_Pf - R_Pr + self.MCP_surf_MCP_vol_ratio * params['PermMCPPropionyl'] * (x[3 + n_compounds_cell] - x[3])  # microcompartment equation for P
        d[4] = self.MCP_surf_MCP_vol_ratio * params['PermMCPPropionate'] * (x[4 + n_compounds_cell] - x[4])  # microcompartment equation for P

        ####################################################################################
        ##################################### cytosol of cell ##############################
        ####################################################################################


        d[5] = - params['PermCellPropanediol'] * self.cell_surf_cell_vol_ratio * (x[5] - x[5 + n_compounds_cell]) \
               - nmcps * params['PermMCPPropanediol'] * self.MCP_surf_cell_vol_ratio * (x[5] - x[5- n_compounds_cell])
        
        d[6] = - params['PermCellPropionaldehyde'] * self.cell_surf_cell_vol_ratio * (x[6] - x[6 + n_compounds_cell]) \
               - nmcps*params['PermMCPPropionaldehyde'] * self.MCP_surf_cell_vol_ratio * (x[6] - x[6- n_compounds_cell])
        
        d[7] = - params['PermCellPropanol'] * self.cell_surf_cell_vol_ratio * (x[7] - x[7 + n_compounds_cell]) \
               - nmcps*params['PermMCPPropanol'] * self.MCP_surf_cell_vol_ratio * (x[7] - x[7- n_compounds_cell])
        
        d[8] = -R_Lf - params['PermCellPropionyl'] * self.cell_surf_cell_vol_ratio * (x[8] - x[8 + n_compounds_cell]) \
               - nmcps*params['PermMCPPropionyl'] * self.MCP_surf_cell_vol_ratio * (x[8] - x[8- n_compounds_cell])
        
        d[9] = R_Lf - params['PermCellPropionate'] * self.cell_surf_cell_vol_ratio * (x[9] - x[9 + n_compounds_cell]) -\
               nmcps * params['PermMCPPropionate'] * self.MCP_surf_cell_vol_ratio * (x[9] - x[9- n_compounds_cell])


        #####################################################################################
        ######################### external volume equations #################################
        #####################################################################################

        d[10] = self.cell_surf_external_vol_ratio * params['PermCellPropanediol'] * ncells * (x[10 - n_compounds_cell] - x[10])  # external equation for concentration
        d[11] = self.cell_surf_external_vol_ratio * params['PermCellPropionaldehyde'] * ncells * (x[11 - n_compounds_cell] - x[11])  # external equation for concentration
        d[12] = self.cell_surf_external_vol_ratio * params['PermCellPropanol'] * ncells * (x[12 - n_compounds_cell] - x[12])  # external equation for concentration
        d[13] = self.cell_surf_external_vol_ratio * params['PermCellPropionyl'] * ncells * (x[13 - n_compounds_cell] - x[13])  # external equation for concentration
        d[14] = self.cell_surf_external_vol_ratio * params['PermCellPropionate'] * ncells * (x[14 - n_compounds_cell] - x[14])  # external equation for concentration
        
        return d

    def _set_symbolic_sderiv(self):
        """
        Generates the symbol differential equation
        """
        x_sp = getattr(self, 'x_sp', None)
        if x_sp is None:
            self._set_symbolic_state_vars()
        self.sderiv_symbolic = self._sderiv(sp.symbols('t'), self.x_sp, self.params_sens_sp_dict)


    def _set_symbolic_sderiv_conc_fun(self):
        """
        Generates the symbol jacobian of the differential equation
        wrt state variables
        """
        sderiv_symbolic = getattr(self, 'sderiv_symbolic', None)
        if sderiv_symbolic is None:
            self._set_symbolic_sderiv()
            sderiv_symbolic = self.sderiv_symbolic
        self.sderiv_jac_conc_sp = sp.Matrix(sderiv_symbolic).jacobian(self.x_sp)
        sderiv_jac_conc_fun_lam = sp.lambdify((sp.symbols('t'),self.x_sp,self.params_sens_sp), self.sderiv_jac_conc_sp, 'numpy')
        self._sderiv_jac_conc_fun = lambda t,x,params_sens_dict: sderiv_jac_conc_fun_lam(t,x,params_sens_dict.values())

    def generate_time_series(self,init_conds, params, n_time_pts = 1000):
        """
        Generates the time series associated with the initial conditions, init_conds,
        and parameters, params
        @param init_conds: dictionary of initial conditions
        @param params: parameters of the system
        @return: time_concat, 1 x n_time_pts, array of time points and
                sol_concat, 15 x n_time_pts, array of concentration at time points
        """

        # initialize OD
        y0 = np.zeros(self.nvars)
        for i, init_names in enumerate(VARIABLE_INIT_NAMES):
            y0[i] = init_conds[init_names]

        ds = lambda t, x: self._sderiv(t, x, params)

        #solve ODE
        sol = solve_ivp(ds, [0, self.fin_exp_time * HRS_TO_SECS], y0, method="LSODA",
                        t_eval=np.linspace(0, self.fin_exp_time * HRS_TO_SECS, n_time_pts),atol=1e-6, rtol=1e-6)

        return sol.t, sol.y.T

