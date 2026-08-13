import numpy as np 
def discrete_moments(values, probabilities):
    """
    Returns: [E_X, E_X2, variance, std_dev] as a list.
    """
    values = np.array(values)
    probabilities = np.array(probabilities)

    expectation = round(np.dot(values, probabilities), 4) 
    e_2 = round(np.dot(values**2, probabilities), 4) 
    e_3 = round( e_2 - expectation**2, 4) 
    return [expectation,e_2, e_3 ,round(np.sqrt(e_3), 4)  ]
    