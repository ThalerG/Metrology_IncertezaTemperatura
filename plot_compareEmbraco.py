import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import odr
from typing import Union, Callable, List, Tuple, Optional

def exp_model(params, x):
    # TODO: Add docstring

    return params[0] + params[1]*np.exp(params[2]*x)

def estimate_model_with_uncertainty(x: Union[np.ndarray, List[float]],
                                    y: Union[np.ndarray, List[float]],
                                    s_x: Optional[Union[float, np.ndarray, List[float]]] = None,
                                    s_y: Optional[Union[float, np.ndarray, List[float]]] = None,
                                    model: Union[str, Callable] = 'lin',
                                    initial_params: Optional[List[float]] = None,
                                    **kwargs
                                    ) -> Tuple[np.ndarray, np.ndarray, odr.Output]:
    
    '''
    Estimates the parameters of a given model with associated uncertainties using Orthogonal Distance Regression (ODR).

    Parameters:
    - x (array-like): Independent variable data.
    - y (array-like): Dependent variable data.
    - s_x (scalar or array-like, optional): Standard deviations of the independent variable data. Default is None.
    - s_y (scalar or array-like, optional): Standard deviations of the dependent variable data. Default is None.
    - model (str or callable, optional): The model to fit. Can be 'lin' for a linear model, 'exp' for an exponential model, or a callable function. Default is 'lin'.
    - initial_params (list): Initial guess for the model parameters. Default is [1.0, 1.0] if model is 'lin' or [1.0, 1.0, 1.0] if model is 'exp'. If model is callable, initial_params must be provided.
    - **kwargs: Additional keyword arguments to be passed to the ODR fitting process.

    Returns:
    - fitted_params (array): Estimated parameters of the model.
    - uncertainty (array): Standard deviations of the estimated parameters.
    - result (odr.Output): Full output of the ODR fitting process, containing additional information such as the residuals and the covariance matrix.

    '''

    if not callable(model):
        if model == 'lin':
            def model(params, x):
                return params[1] * x + params[0]
            initial_params = [1.0, 1.0] if initial_params is None else initial_params

        elif model == 'exp':
            def model(params, x):
                return params[0] + params[1] * np.exp(params[2] * x)
            initial_params = [1.0, 1.0, 1.0] if initial_params is None else initial_params

        elif not callable(model):
            raise ValueError("Model must be a callable function or 'lin' or 'exp'.")
        
        elif callable(model) and initial_params is None:
            raise ValueError("Initial parameters must be provided when using a custom model.")

    # Define o modelo
    model = odr.Model(model)

    if s_x == 0:
        s_x = None
    if s_y == 0:
        s_y = None

    # Create objeto RealData
    data = odr.RealData(x, y, sx = s_x, sy = s_y)

    # ODR fit
    odr_fit = odr.ODR(data, model, beta0=initial_params, **kwargs)
    result = odr_fit.run()

    # Fitted parameters
    fitted_params = result.beta

    # Desvio padrão dos parâmetros estimados
    uncertainty = result.sd_beta

    return fitted_params, uncertainty, result

def get_unc_GUM(baseValues, x_og, y_og):
    
    initial_params = [8.50468513, 0.03510031, -3.936979] 

    N1 = baseValues['N1']
    dN = baseValues['dN']
    n_x =  baseValues['Npoints']

    ind = range(max(N1,0), min(N1 + n_x * dN,len(x_og)), dN)

    s_t0 = baseValues['s_t0']
    s_dt = baseValues['s_dt']
    s_dR = baseValues['s_dR']

    s_R1 = baseValues['s_R1']
    s_Tamb1 = baseValues['s_Tamb1']
    s_Tamb2 = baseValues['s_Tamb2']
    s_cvol = 1

    x_tot = x_og[ind]
    y_tot = y_og[ind]

    params, _, res = estimate_model_with_uncertainty(x_tot, y_tot, s_dt, s_dR, model=exp_model, initial_params= initial_params,maxit = 1000000)

    B0 = params[0]
    B1 = params[1]
    B2 = params[2]

    s_B0 = np.sqrt(res.cov_beta[0][0])
    s_B1 = np.sqrt(res.cov_beta[1][1])
    s_B2 = np.sqrt(res.cov_beta[2][2])

    Tamb_1 = baseValues['Tamb_1']
    Tamb_2 = baseValues['Tamb_2']
    R1 = baseValues['R1']
    cvol = 100
    t0 = 0

    k = 25450/cvol-20

    R2 = B0+B1*np.exp(B2*t0)

    s_DT_cov = [((R2-R1)/R1 + 1)*s_Tamb1, # Uncertainty of initial ambient temperature
            -1*s_Tamb2, # Uncertainty of final ambient temperature
            -R2*(k+Tamb_1)/(R1**2)*s_R1, # Uncertainty of initial resistance
            -(R2-R1)/R1*25450/cvol**2*s_cvol, # Uncertainty of copper purity
            -R2*(k+Tamb_1)/(R1**2)*s_B0, # Uncertainty of beta0
            -np.exp(B2*t0)*R2*(k+Tamb_1)/(R1**2)*s_B1, # Uncertainty of beta1
            -t0*B1*np.exp(B2*t0)*R2*(k+Tamb_1)/(R1**2)*s_B2, # Uncertainty of beta2
            -B2*B1*np.exp(B2*t0)*R2*(k+Tamb_1)/(R1**2)*s_t0 # Uncertainty of t0
            ]
    
    s = []
    for sign0 in [True,False]:
        b0 = (B0+s_B0) if sign0 else (B0-s_B0)
        for sign1 in [True,False]:
            b1 = (B1+s_B1) if sign1 else (B1-s_B1)
            for sign2 in [True,False]:
                b2 = (B2+s_B2) if sign2 else (B2-s_B2)
                for signt0 in [True,False]:
                    t0 = (t0+s_t0) if signt0 else (t0-s_t0)
                    R = b0+b1*np.exp(b2*t0)-R2
                    s.append(abs(R))
    sR2_bound = np.max(s)

    s_DT_bound = np.linalg.norm(s_DT_cov[0:4] + [sR2_bound])
    
    DT = (R2-R1)/R1*(k+Tamb_1)-(Tamb_2-Tamb_1)

    s_noR2 = np.linalg.norm(s_DT_cov[0:4])
    return DT, np.linalg.norm(s_DT_cov), s_DT_cov, s_DT_bound, s_noR2

def analysis_fit(ax,comp,nLim=9):
    if comp == 1:
        file_path = "Dados/comp1.csv"
    elif comp == 2:
        file_path = "Dados/comp2.csv"

    df = pd.read_csv(file_path)

    x_og = df['Time'].values[:nLim]
    y_og = df['Resistance'].values[:nLim]

    


if __name__ == "__main__":
    cm = 1/2.54 
    fig, (ax_top, ax_bottom) = plt.subplots(2, 2, figsize=(22*cm, 11*cm))

    for ax in ax_bottom:
        fig.delaxes(ax)

    ax_bottom = plt.subplot(2, 1, 2)

    baseValues = {'dN': [1],
            'Npoints': [9],
            'N1': [0],
            'R1': [1],
            't0': [0],
            'Tamb1': [25],
            'Tamb2': [25],
            'cvol': [100],
            's_t0': [0.1],
            's_dt': [0.001],
            's_dR': [0.001],
            's_R1': [0.001],
            's_Tamb1': [0.2],
            's_Tamb2': [0.2],
            's_cvol': [1]}

    analysis_fit(axtop[0],comp1)
