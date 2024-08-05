import numpy as np
from scipy import odr

def estimate_model_with_uncertainty(x, y, s_x=None, s_y=None, model='lin', initial_params=None):
    '''
    Estimates the parameters of a given model with associated uncertainties using Orthogonal Distance Regression (ODR).

    Parameters:
    - x (array-like): Independent variable data.
    - y (array-like): Dependent variable data.
    - s_x (scalar or array-like, optional): Standard deviations of the independent variable data. Default is None.
    - s_y (scalar or array-like, optional): Standard deviations of the dependent variable data. Default is None.
    - model (str or callable, optional): The model to fit. Can be 'lin' for a linear model, 'exp' for an exponential model, or a callable function. Default is 'lin'.
    - initial_params (list): Initial guess for the model parameters. Default is [1.0, 1.0] if model is 'lin' or [1.0, 1.0, 1.0] if model is 'exp'. If model is callable, initial_params must be provided.

    Returns:
    - fitted_params (array): Estimated parameters of the model.
    - uncertainty (array): Standard deviations of the estimated parameters.
    - result (odr.Output): Full output of the ODR fitting process, containing additional information such as the residuals and the covariance matrix.

    '''

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

    # Create objeto RealData
    data = odr.RealData(x, y, sx = s_x, sy = s_y)

    # ODR fit
    odr_fit = odr.ODR(data, model, beta0=initial_params)
    result = odr_fit.run()

    # Fitted parameters
    fitted_params = result.beta

    # Desvio padrão dos parâmetros estimados
    uncertainty = result.sd_beta

    return fitted_params, uncertainty, result

if __name__ == "__main__":
    # Parâmetros iniciais
    initial_params = [2.0, 1.0]

    # Dados
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([1.1, 2.1, 3.1, 4.1, 5.1])

    # Incerteza dos dados
    s_x = np.array(0.1)
    s_y = np.array(0.1)

    # Estimação
    fitted_params, uncertainty, result = estimate_model_with_uncertainty(x, y, s_x, s_y, 'lin', initial_params)

    print("Uncertainty:", uncertainty)
    print("Slope:", fitted_params[1])
    print("Intercept:", fitted_params[0])