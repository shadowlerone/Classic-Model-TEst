import numpy as np
import matplotlib.pyplot as plt

def fplot(litr_output,soil_output,resp_output,site):
    '''
    Function that plots model outputs
    :param output: Model output, no formatting needed
    :return:
    '''
    x = np.arange(0,litr_output.shape[2])

    # Formating outputs
    litr_format = np.sum(litr_output,axis=(0,1))
    soil_format = np.sum(soil_output,axis=(0,1))
    resp_format = round(np.mean(np.sum(resp_output, axis=(0,1,2))), 3)

    # plotting results
    plt.plot(x, litr_format, '-b', label='Litter pool')
    plt.plot(x, soil_format, '-r', label='Soil pool')
    plt.axhline(resp_format, ls=':', c='g')
    plt.text(20000,10,'Average repiration:'+ str(resp_format))

    # Formatting plot
    plt.title('Soil carbon simulation of fluxnet site '+ site)
    plt.ylabel('Total carbon content [kg C $m^{-2}$]')
    plt.xlabel('Number of days simulated')
    plt.legend()
    plt.show()