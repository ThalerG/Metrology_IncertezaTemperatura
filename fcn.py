import numpy as np
from scipy import odr
from typing import Union, Callable, List, Tuple, Optional

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


def final_temperature(R1:float, R2:float, T_amb1:float, T_amb2:float ,k:float) -> float:
    """
    Calculates the final temperature T2 based on the resistance values R2 and R1, the ambient temperatures T_amb1 and T_amb2, and the reciprocal of the temperature coefficient k.
    Parameters:
    - R1 (float): Initial resistance.
    - R2 (float): Final resistance.
    - T_amb1 (float): Ambient temperature at the beginning of the test.
    - T_amb2 (float): Ambient temperature at the end of the test.
    - k (float): Reciprocal of the temperature coefficient at 0°C.

    - T2 (float): Calculated final temperature T2.
    """
    return (R2 - R1) * (k + T_amb1) / (R1) - (T_amb2 - T_amb1) + T_amb1

def delta_temperature(R1:float, R2:float, T_amb1:float, T_amb2:float ,k:float) -> float:
    """
    Calculates the final temperature T2 based on the resistance values R2 and R1, the ambient temperatures T_amb1 and T_amb2, and the reciprocal of the temperature coefficient k.
    Parameters:
    - R1 (float): Initial resistance.
    - R2 (float): Final resistance.
    - T_amb1 (float): Ambient temperature at the beginning of the test.
    - T_amb2 (float): Ambient temperature at the end of the test.
    - k (float): Reciprocal of the temperature coefficient at 0°C.

    - T2 (float): Calculated final temperature T2.
    """
    return (R2 - R1) * (k + T_amb1) / (R1) - (T_amb2 - T_amb1)

def final_temperature_uncertainty(R1:float, R2:float, T_amb1:float, T_amb2:float, k:float, s_R1:float, s_R2:float, s_T_amb1:float, s_T_amb2:float) -> float:
    """
    Calculates the combined uncertainty of temperature (s_T2) based on the given parameters.
    
    Parameters:
    - R1 (float): Initial resistance value.
    - R2 (float): Final resistance value.
    - T_amb1 (float): Ambient temperature at the beginning of the test.
    - T_amb2 (float): Ambient temperature at the end of the test.
    - k (float): Reciprocal of the temperature coefficient at 0°C.
    - s_R1 (float): Uncertainty of initial resistance.
    - s_R2 (float): Uncertainty of final resistance.
    - s_T_amb1 (float): Uncertainty of initial ambient temperature.
    - s_T_amb2 (float): Uncertainty of final ambient temperature.
    
    Returns:
    - s_T2 (float): Combined uncertainty of temperature.
    
    Notes:
    - The combined uncertainty is calculated as the square root of the sum of squares of individual uncertainties.
    - The individual uncertainties are derived from the partial derivatives of the temperature equation with respect to each parameter. The original equation is:
        T2 = (R2 - R1) * (k + T_amb1) / R1 - (T_amb2 - T_amb1) + T_amb1
    """
    s_T2 = [((R2-R1)/R1 + 2)*s_T_amb1, # Uncertainty of initial ambient temperature
            -1*s_T_amb2, # Uncertainty of final ambient temperature
            -R2*(k+T_amb1)/(R1**2)*s_R1, # Uncertainty of initial resistance
            (k+T_amb1)/R1*s_R2] # Uncertainty of final resistance
    
    return np.linalg.norm(s_T2)

def delta_temperature_uncertainty(R1:float, R2:float, T_amb1:float, T_amb2:float, k:float, s_R1:float, s_R2:float, s_T_amb1:float, s_T_amb2:float) -> float:
    """
    Calculates the combined uncertainty of temperature (s_T2) based on the given parameters.
    
    Parameters:
    - R1 (float): Initial resistance value.
    - R2 (float): Final resistance value.
    - T_amb1 (float): Ambient temperature at the beginning of the test.
    - T_amb2 (float): Ambient temperature at the end of the test.
    - k (float): Reciprocal of the temperature coefficient at 0°C.
    - s_R1 (float): Uncertainty of initial resistance.
    - s_R2 (float): Uncertainty of final resistance.
    - s_T_amb1 (float): Uncertainty of initial ambient temperature.
    - s_T_amb2 (float): Uncertainty of final ambient temperature.
    
    Returns:
    - s_T2 (float): Combined uncertainty of temperature.
    
    Notes:
    - The combined uncertainty is calculated as the square root of the sum of squares of individual uncertainties.
    - The individual uncertainties are derived from the partial derivatives of the temperature equation with respect to each parameter. The original equation is:
        T2 = (R2 - R1) * (k + T_amb1) / R1 - (T_amb2 - T_amb1) + T_amb1
    """
    s_T2 = [((R2-R1)/R1 + 1)*s_T_amb1, # Uncertainty of initial ambient temperature
            -1*s_T_amb2, # Uncertainty of final ambient temperature
            -R2*(k+T_amb1)/(R1**2)*s_R1, # Uncertainty of initial resistance
            (k+T_amb1)/R1*s_R2] # Uncertainty of final resistance
    
    return np.linalg.norm(s_T2)

def generate_estimation_models(type: str ='poly', degree:int = 1, params: List[float] = None) -> Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]]:
    """
    Returns a model function based on the specified type and degree.
    Parameters:
    - type (str): The type of model to be returned. Valid options are 'poly' and 'exp'. Default is 'poly'.
    - degree (int): The degree of the polynomial model. Only applicable when type is 'poly'. Default is 1.
    - params (List[float]): List of parameters for the model. If provided, the returned model function will use these parameters.
    Returns:
    - model (function): The model function that takes parameters and input values and returns the estimated output.
    Example usage:
    >>> poly_model = estimation_models('poly', 2)
    >>> poly_model([1, 2, 3], 4)
    27
    >>> exp_model = estimation_models('exp')
    >>> exp_model([1, 2, 3], 4)
    53.598150033144236
    """

    if params is None: # If no parameters are provided, return a model function that takes parameters as input
        if type == 'poly':
            def model(parameters, x):
                pow = np.array(range(degree+1))
                return sum([parameters[i]*x**i for i in pow])
        
        if type == 'exp':
            def model(parameters, x):
                return parameters[0] + parameters[1] * np.exp(parameters[2] * x)
        
        return model

    else: # If parameters are provided, return a model function that uses the provided parameters
        if type == 'poly':

            if len(params) != degree + 1:
                raise ValueError("The number of elements in params for exponential functions must be equal to 3.")
        
            def model(x):
                pow = np.array(range(degree+1))
                return sum([params[i]*x**i for i in pow])
        
        if type == 'exp':
            if len(params) != 3:
                raise ValueError("The number of elements in params for exponential functions must be equal to 3.")
            
            def model(x):
                return params[0] + params[1] * np.exp(params[2] * x)
        
        return model

    
    
def generate_estimation_uncertainty_models(params: List[float], s_params: List[float], s_x: float, type: str ='poly', degree: int = 1) -> Callable[[Union[float, np.ndarray]], float]:
    """
    Returns a model function that calculates the estimation uncertainty based on the given parameters.
    Parameters:
    - params (List[float]): List of parameters for the model.
    - s_params (List[float]): List of uncertainties for the parameters.
    - s_x (float): Uncertainty of x.
    - type (str, optional): Type of model to use. Defaults to 'poly'.
    - degree (int, optional): Degree of the polynomial model. Defaults to 1.
    Returns:
    - model (Callable[[Union[float, np.ndarray]], float]): Model function that calculates the estimation uncertainty. Works for array or scalar.
    Raises:
    - None
    Example usage:
    >>> params = [1, 2, 3]
    >>> s_params = [0.1, 0.2, 0.3]
    >>> s_x = 0.01
    >>> model_func = estimation_uncertainty_models(params, s_params, s_x, type='poly', degree=2)
    >>> uncertainty = model_func(5)
    ```    
    """

    if type == 'poly':
        if len(params) != degree + 1:
            raise ValueError("The number of elements in params for polynomial functions must be equal to degree + 1.")
        
        def model(x: Union[float, np.ndarray]):
            if hasattr(x, "__iter__"):
                return [model(x_el) for x_el in x]
            else:
                pow = np.array(range(degree+1))
                s_model = x**pow * s_params # Uncertainty for each polynomial term beta_i is x**i * s_beta_i
                np.append(s_model, np.sum(pow[1:]*params[1:]*x**(pow[1:]-1))*s_x) # Uncertainty of x

            return np.linalg.norm(s_model) # Combined uncertainty is the square root of the sum of squares
            
    if type == 'exp':
        if len(params) != 3:
            raise ValueError("The number of elements in params for exponential functions must be equal to 3.")
        
        def model(x: Union[float, np.ndarray]):
            if hasattr(x, "__iter__"):
                return [model(x_el) for x_el in x]
            else:
                s_model = [1*s_params[0], # Uncertainty of parameter 0
                    np.exp(params[2]*x)*s_params[1], # Uncertainty of parameter 1
                    x*params[1]*np.exp(params[2]*x)*s_params[2], # Uncertainty of parameter 2
                    params[1]*params[2]*np.exp(params[2]*x)*s_x, # Uncertainty of x
                    ]
                return np.linalg.norm(s_model)
        
    return model

def get_formula(type: str ='poly', degree: int = 1, params: List[float] = None) -> str:
    """
    Returns a HTML string representation of the formula of the model based on the specified type and degree.
    Parameters:
    - type (str): The type of model. Valid options are 'poly' and 'exp'. Default is 'poly'.
    - degree (int): The degree of the polynomial model. Only applicable when type is 'poly'. Default is 1.
    - params (List[float]): List of parameters for the model. If provided, the returned formula will use these parameters.
    Returns:
    - formula (str): The string representation of the model formula.
    Example usage:
    >>> get_formula('poly', 2)
    'a<sub>0</sub> + a<sub>1</sub>x + a<sub>2</sub>x<sup>2</sup>'
    >>> get_formula('exp')
    'a<sub>0</sub> + a<sub>1</sub>e<sup>a<sub>2</sub>x</sup>'
    """
    if type == 'poly':
        for i in range(degree+1):
            if params is None:
                if i == 0:
                    formula = f'a<sub>{i}</sub>'
                else:
                    formula += f'+ a<sub>{i}</sub>x<sup>{i}</sup>'
            else:
                if i == 0:
                    formula = f'{params[i]:.5g}'
                else:
                    if params[i] < 0:
                        formula += f'- {abs(params[i]):.3g}x'
                    else:
                        formula += f'+ {params[i]:.3g}x'
                    
                    if i > 1:
                        formula += f'<sup>{i}</sup>'

                
        return formula
    
    if type == 'exp':
        if params is None:
            formula = 'a<sub>0</sub> + a<sub>1</sub>e<sup>a<sub>2</sub>x</sup>'
        else:
            if params[1] < 0:
                formula = f'{params[0]:.5g} - {abs(params[1]):.3g}*e<sup>{params[2]:.3g}x</sup>'
            else:
                formula = f'{params[0]:.5g} + {params[1]:.3g}*e<sup>{params[2]:.3g}x</sup>'
        return formula

    raise ValueError("Invalid model type. Must be 'poly' or 'exp'.")

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