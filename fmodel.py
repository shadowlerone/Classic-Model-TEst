import numpy as np
import fenv
import turbationf
from alive_progress import alive_bar

def model(apfts, nruns, iccp2, ignd, litrmass, soilcmas, ncell, tf, delt, ndays, thice, thpor, thliq, psisat, isand,
		  tbar, b, itermax, spinup, zbotw, actlyr, SAND, fcancmx, delzw, root_input, fLeafLitter, fStemLitter):
	"""
	Function that performs the simulation using the CLASSIC soil component. Simulates 1 or more specific sites
	:param apfts: pft specific parameter array
	:param nruns: number of different sites to simulate in a single run
	:param iccp2: number of pfts, 9 ctem pfts + bare + LUC
	:param ignd: number of soil layers
	:param litrmass: initial mass of carbon in the litter pool [kg C m^-2] (ignd, iccp1, nruns)
	:param soilcmas: initial mass of carbon in the soil pool [kg C m-2] (ignd, iccp1, nruns)
	:param ncell: number of gridcells to simulate
	:param tf: length of available data to drive model [yr]
	:param delt: length of one iteration [yr]
	:param ndays: number of days of available data [days]
	:param thice: Frozen water  content at each soil layer [m^3 m^-3] (ndays, ignd, nruns)
	:param thpor: Soil total porority [m^3 m^-3] (ignd, nruns)
	:param thliq: Liquid water content at each soil layer [m^3 m^-3] (ndays, ignd, nruns)
	:param psisat: Soil moisture suction at saturation [m] (ignd, nruns)
	:param isand: Soil content flag -4 = ice, -3 = bedrock
	:param tbar: Temperature at each soil layer [K] (ndays, ignd, nruns)
	:param b: Clapp and Hornberger empirical param [] (ignd, nruns)
	:param itermax: maximum number of iteration to simulate
	:param spinup: number of spinup iterations
	:param zbotw: Bottom of permeable soil layers [m] (ignd, nruns)
	:param actlyr: Active layer depth [m] (ndays, nruns)
	:param SAND: Sand content of soil [%] (ignd, nruns)
	:param fcancmx: pft coverage of gridcells [%] (iccp2, nruns)
	:param delzw: thickness of permeable part of soil layer [m] (ignd, nruns)
	:param root_input: Total Carbon Mass Flux from Roots to Litter [kg m^-2 s^-1]
	:param fLeafLitter: Total carbon mass flux from leaf to litter [kg m^-2 s^-1]
	:param fStemLitter: Total carbon mass flux from stem to litter [kg m^-2 s^-1]
	:return:
	"""

	# Initializing soil C content vector
	litrplsz = np.zeros([nruns, iccp2, ignd])
	soilplsz = np.zeros([nruns, iccp2, ignd])
	litrplsz += litrmass[:, :iccp2].reshape(iccp2, ignd)  # initial carbon mass of litter pool [kg C / m^2]
	soilplsz += soilcmas[:, :iccp2].reshape(iccp2, ignd)  # initial carbon mass of soil pool [kg C / m^2]

	litrplsz = litrplsz.transpose(1, 2, 0)
	soilplsz = soilplsz.transpose(1, 2, 0)

	# Initializing litter input array
	Cinput = np.zeros([iccp2, ignd, ncell])  # Pool carbon input

	# arrays for outputs
	litter = np.empty([ignd, nruns, itermax])
	soil = np.zeros([ignd, nruns, itermax])
	resp = np.zeros([ignd, iccp2, nruns, itermax])

	count = 0  # iteration count
	bsratelt = apfts[0, :]  # turnover rate of litter pool [kg C/kg C.yr]
	bsratesc = apfts[1, :]  # turnover rate of soil pool [kg C/kg C.yr]
	humicfac = apfts[2, :]  # Humification factor
	kterm = 3  # constant that determines the depth at which turbation stops
	biodiffus = 2.73972e-07  # Diffusivity coefficient for biodiffus [m^2 / day]
	cryodiffus = 1.36986e-06  # Diffusivity coefficient for cryodiffus [m^2 / day]
	r_depthredu = 10  # depth reduction parameter


	reduceatdepth = np.exp(-delzw / r_depthredu)  # Depth reduction factor

	sample = int(4 / delt)
	# Computing env. modifiers beforehand
	envmod = fenv.fenv(ndays, thice, thpor, thliq, psisat, ignd, isand, tbar, b, nruns)
	print('Envmod successfully calculated')
	envmodltr = envmod[0]
	envmodsol = envmod[1]
	# Time loop
	print('Starting model run')
	with alive_bar(itermax, bar='smooth', spinner='waves2') as bar:  # Setting up progress bar
		while count < itermax:

			t = count % sample  # cycling on the length of imput files
			# Computing environmental modifier at current timestep
			envltr = envmodltr[t]  # Litter pool environmental modifier at current timestep
			envsol = envmodsol[t]  # Soil pool environmental modifier at current timestep

			# Modifying transfert coeff if spinup=True
			if count <= spinup:
				spinfast = 10  # transfer coeff from litter pool to soil
			else:
				spinfast = 1

			# Computing input vector at current time step
			Cinput[:, 0, 0] = (fLeafLitter[t] + fStemLitter[t])  # Carbon input to litter pool at timestep t [Kg C m^-2 yr^-1]
			#Cinput[:, :, 0] += root_input[t]  # Carbon input from roots at timestep t [Kg C m^-2 yr^-1]

			# Computing respiration over time step for both pool
			ltresveg = bsratelt * (litrplsz * envltr * reduceatdepth).transpose(1, 2, 0)  # [kg C m^-2 yr^-1]
			scresveg = bsratesc * (soilplsz * envsol * reduceatdepth).transpose(1, 2, 0)  # [kg C m^-2 yr^-1]

			# Computing pool size change in both pool
			dx1 = Cinput - (ltresveg * (1 + humicfac)).transpose(2, 0, 1)
			dx2 = spinfast * ((ltresveg * humicfac) - scresveg).transpose(2, 0, 1)

			# Updating pools
			litrplsz += dx1 * delt  # Updating litter pool [kg C m^-2]
			soilplsz += dx2 * delt  # Updating soil pool [kg C m^-2]

			# Calling turbation subroutine
			litrplsz, soilplsz = turbationf.turbation(
				litrplsz, 
				soilplsz, 
				zbotw, 
				ignd, 
				kterm, 
				actlyr[t], 
				cryodiffus,
				biodiffus, 
				spinfast, 
				SAND, 
				nruns, 
				delt, 
				iccp2)

			gridltr = np.sum(litrplsz.transpose(1, 0, 2) * fcancmx, axis=1)  # Litter pool size at grid level
			gridsol = np.sum(soilplsz.transpose(1, 0, 2) * fcancmx, axis=1)  # SoilC pool size at grid level
			gridrsp = np.sum((ltresveg + scresveg) * fcancmx, axis=1)  # Respiration at grid level
			# Adding to output arrays
			litter[:, :, count] = gridltr
			soil[:, :, count] = gridsol
			resp[:, :, 0, count] = gridrsp

			count += 1  # Iteration count
			bar()  # Update progress bar

	return litter, soil, resp