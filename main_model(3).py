import numpy as np
import xarray as xr
import frmatctem
import fmodel
import fplot


######## ENVIRONMENT VARIABLES TWEAK ########
temp_var = 0.0
moist_var = 0.0
#############################################

# Importing model run params
my_dir = '/home/charlesgauthier/project/' # Directory in which files are contained
site = 'FI-Hyy'

if site == 'FI-Hyy':
    tf = 19
    ndays = tf * 365
if site == 'GH-Ank':
    tf = 4
    ndays = tf * 365
if site == 'CA-TVC':
    tf = 7
    ndays = tf *365 + 1

nruns = 1
iccp2 = 11  # Number of pfts


# Loading input files
init = xr.open_dataset(my_dir + site + '/'+site+ '_init.nc')
mrsfl = xr.open_dataset(my_dir+site+'/mrsfl_daily.nc')
mrsll = xr.open_dataset(my_dir+site+'/mrsll_daily.nc')
tsl = xr.open_dataset(my_dir+site+'/tsl_daily.nc')
fLeafLitter = xr.open_dataset(my_dir+site+'/fLeafLitter_daily_perpft.nc')
fRootLitter = xr.open_dataset(my_dir+site+'/fRootLitter_daily_perpft.nc')
fStemLitter = xr.open_dataset(my_dir+site+'/fStemLitter_daily_perpft.nc')
actlyr = xr.open_dataset(my_dir+site+'/actlyr_daily.nc')
cRoot = xr.open_dataset(my_dir+site+'/cRoot_daily_perpft.nc')
print('Input files loaded successfully')

ignd = init.dims['layer'] # Number of soil layers
ncell = 1  # Number of gridcells

fLeafLitter = fLeafLitter['fLeafLitter'].data[:, :iccp2, 0, 0]      # Total carbon input from leaf to litter pool
fLeafLitter = np.reshape(fLeafLitter, (ndays, iccp2))    # Reshaping and converting from s^-1 to yr^-1

fRootLitter = fRootLitter['fRootLitter'].data[:, :iccp2, 0, 0]      # Total carbon input from Root to litter pool
fRootLitter = np.reshape(fRootLitter, (ndays, iccp2))    # Reshaping and converting from s^-1 to yr^-1

fStemLitter = fStemLitter['fStemLitter'].data[:, :iccp2, 0, 0]      # Total carbon input from stem to litter pool
fStemLitter = np.reshape(fStemLitter, (ndays, iccp2))    # Reshaping and converting from s^-1 to yr^-1

fcancmx = init['fcancmx'].data[0, :iccp2, 0, 0]                     # Fraction of pft coverage
fcancmx = np.tile(fcancmx, (nruns, 1)).transpose()                  # Reshaping

# Global variables
# Reading and formatting variables from initialisation file
SAND = init['SAND'].data[0, :, 0, 0]  # Sand content of soil [%]
CLAY = init['CLAY'].data[0, :, 0, 0]  # Clay content of soil [%]

# Reshaping SAND
temp = np.zeros([ignd, ncell])
temp[:, 0] = SAND
SAND = temp

# Reshaping CLAY
temp = np.zeros([ignd, ncell])
temp[:, 0] = CLAY
CLAY = temp

sdepth = init['SDEP'].data[0, 0, :]  # Soil Permeable depth [m]
maxAnnualActLyr = init['maxAnnualActLyr'].data[0, 0, 0]  # Active layer thickness max over e-folding period

# Initial state of carbon pools
litrmass = init['litrmass'].data[0, :, :iccp2, 0, 0]  # Mass of C in litter pool [kg C m^-2]
temp = np.zeros([ignd, iccp2, ncell])
temp[:, :, 0] = litrmass
litrmass = temp

soilcmas = init['soilcmas'].data[0, :, :iccp2, 0, 0]  # Mass of C in soil pool [kg C m^-2]
temp = np.zeros([ignd, iccp2, ncell])
temp[:, :, 0] = soilcmas
soilcmas = temp

# Calculating delzw and zbotw
isand = np.where((SAND == -4) | (SAND == -3))  # Flag for soil type -4 = ice, -3 = rock
delz = init['DELZ'].data[:]  # Ground layer thickness [m]
temp = np.zeros([ignd, ncell])
temp[:, 0] = delz
delz = temp
zbot = np.zeros_like(delz)  # Dept of the bottom of each soil layer [m]
for i in range(0, len(delz)):
    zbot[i] = np.sum(delz[:i + 1])

delzw = np.zeros_like(delz)  # thickness of permeable part of soil layer
isice = np.where(SAND == -4)  # Soil layers that are ice
isrock = np.where(SAND == -3)  # Soil layers that are rock
isbelow = np.where(sdepth >= zbot)  #
isnear = np.where(sdepth < (zbot - delz + 0.025))

if SAND[0] == -4:  # is ice
    delzw = delz
    SAND[:] = -4
else:
    delzw = np.maximum(0.05, (sdepth - (zbot - delz)))

    delzw[isnear] = 0.0
    SAND[isnear] = -3

    delzw[isbelow] = delz[isbelow]
    delzw[isrock] = 0

zbotw = np.maximum(0, (zbot - delz)) + delzw  # Bottom of soil layers [m]
isand = np.where((SAND == -4) | (SAND == -3))  # Flag ,-4 = ice, -3 = ice


# Soil properties variables
mliq = mrsll['mrsll'].data[:, :, 0, 0]  # Mass of liquid water at each soil layer [kg m^-2]
temp = np.zeros([ndays, ignd, 1])
temp[:, :, 0] = mliq
mliq = temp
thliq = mliq / 1000 / delzw + moist_var # Conversion to m^3/m^3

mice = mrsfl['mrsfl'].data[:, :, 0, 0]  # Mass of frozen water at each soil layer [kg m^-2]
temp = np.zeros([ndays, ignd, 1])
temp[:, :, 0] = mice
mice = temp
thice = mice / 1000 / delzw  # Conversion to m^3/m^3

tbar = tsl['tsl'].data[:, :, 0, 0] + temp_var  # Temperature at each soil layer [K]
temp = np.zeros([ndays, ignd, 1])
temp[:, :, 0] = tbar
tbar = temp
thpor = (-0.126 * SAND + 48.9) / 100  # Soil total porority [m^3 m^-3]
thpor[isand] = 0  # No value where soil is rock or ice
b = 0.159 * CLAY + 2.91  # Clapp and Hornberger empirical param []
b[isand] = 0  # No value where soil is rock or ice
psisat = 0.01 * np.exp(-0.0302 * SAND + 4.33)  # Soil moisture suction at saturation [m]
psisat[isand] = 0

# Array of pft specific variables, 9 CTEM pfts + Bare + LUC
apfts = np.array(
    [[0.4453, 0.5986, 0.6339, 0.7576, 0.6957, 0.6000, 0.6000, 0.5260, 0.5260, 0.5605, 0.5605],  # 0 bsratelt
     [0.0260, 0.0260, 0.0208, 0.0208, 0.0208, 0.0350, 0.0350, 0.0125, 0.0125, 0.02258, 0.02258],  # 1 bsratesc
     [0.42, 0.42, 0.53, 0.48, 0.48, 0.10, 0.10, 0.42, 0.42, 0.45, 0.45],  # 2 humicfac
     [4.70, 5.86, 3.87, 3.46, 3.97, 3.97, 3.97, 5.86, 4.92, 4.0, 4.0],  # 3 abar
     [1.85, 1.45, 2.45, 2.10, 2.10, 0.10, 0.10, 0.70, 0.70, 0.0, 0.0],  # 4 avertmas
     [0.80, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80, 0.0, 0.0],  # 5 alpha
     [3.00, 3.00, 5.00, 5.00, 3.00, 2.00, 2.00, 1.00, 1.00, 0.0, 0.0]])  # 6 mxrtdpth

abar = apfts[3, :]      # Parameter determining average root profile
avertmas = apfts[4, :]  # average root biomass [kg C m^-2]
alpha = apfts[5, :]
mxrtdpth = apfts[6, :]

# Computing carbon input from roots
rmatctem = frmatctem.rmatctemf(zbotw, alpha, cRoot.cRoot.data[:, :11, 0, 0], avertmas, abar, sdepth, maxAnnualActLyr,
                               mxrtdpth)
root_input = (fRootLitter * rmatctem.transpose(2, 0, 1)).transpose(1, 2, 0)

itermax = 80000

delt = 1 / 365
spinup = 100

litr_output, soil_output, resp_output = fmodel.model(apfts, nruns, iccp2, ignd, litrmass, soilcmas, ncell, tf, delt, ndays,
                                              thice, thpor, thliq, psisat, isand, tbar, b, itermax, spinup, zbotw,
                                              actlyr.actlyr.data[:, :, 0], SAND, fcancmx, delzw, root_input,
                                              fLeafLitter, fStemLitter)
fplot.fplot(litr_output,soil_output,resp_output,site)
#np.save('litr_output',litr_output)
#np.save('soil_output', soil_output)
#np.save('resp_output', resp_output)


dummy = 1
