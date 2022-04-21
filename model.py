import numpy as np
import xarray as xr
import frmatctem
import fmodel
import fplot
import tests
import numpy as np
import fenv
import turbationf
from alive_progress import alive_bar
from os import path, mkdir
import test
import matplotlib.pyplot as plt
import pandas as pd
from cachetools import cached, LRUCache, TTLCache

@cached(cache={})
def load_dataset(directory, site):
	init = xr.open_dataset(path.join(directory,site,f"{site}_init.nc"))
	mrsfl = xr.open_dataset(path.join(directory,site,'mrsfl_daily.nc'))
	mrsll = xr.open_dataset(path.join(directory,site,'mrsll_daily.nc'))
	tsl = xr.open_dataset(path.join(directory, site,'tsl_daily.nc'))
	fLeafLitter = xr.open_dataset(path.join(directory, site, 'fLeafLitter_daily_perpft.nc'))
	fRootLitter = xr.open_dataset(path.join(directory, site, 'fRootLitter_daily_perpft.nc'))
	fStemLitter = xr.open_dataset(path.join(directory, site,'fStemLitter_daily_perpft.nc'))
	actlyr = xr.open_dataset(path.join(directory, site,'actlyr_daily.nc'))
	cRoot = xr.open_dataset(path.join(directory, site,'cRoot_daily_perpft.nc'))
	return [init, mrsfl, mrsll, tsl, fLeafLitter, fRootLitter, fStemLitter, actlyr, cRoot]

class Model:
	fp = ''
	def __init__(self, temp_var:float=0.0, moist_var:float=0.0, site:str='FI-Hyy', nruns:int=1, iccp2:int=11, directory:str='', itermax=80000*2): 
		self.my_dir = directory
		self.site = site
		tf = 0
		ndays = tf * 365
		if site == 'FI-Hyy':
			tf = 19
			ndays = tf * 365
		if site == 'GH-Ank':
			tf = 4
			ndays = tf * 365
		if site == 'CA-TVC':
			tf = 7
			ndays = tf *365 + 1
		self.itermax = itermax
		self.nruns = nruns
		self.iccp2 = iccp2
		self.tf = tf
		self.ndays = ndays
		self.temp_var = temp_var
		self.moist_var = moist_var
		# print(self.my_dir)
		# dir_prefix = path.join(self.my_dir,self.site)
		self.init, self.mrsfl, self.mrsll, self.tsl, self.fLeafLitter, self.fRootLitter, self.fStemLitter, self.actlyr, self.cRoot = load_dataset(self.my_dir, self.site)
		# self.init = xr.open_dataset(path.join(self.my_dir,self.site,f"{self.site}_init.nc"))
		# self.mrsfl = xr.open_dataset(path.join(self.my_dir,self.site,'mrsfl_daily.nc'))
		# self.mrsll = xr.open_dataset(path.join(self.my_dir,self.site,'mrsll_daily.nc'))
		# self.tsl = xr.open_dataset(path.join(self.my_dir, self.site,'tsl_daily.nc'))
		# self.fLeafLitter = xr.open_dataset(path.join(self.my_dir, self.site, 'fLeafLitter_daily_perpft.nc'))
		# self.fRootLitter = xr.open_dataset(path.join(self.my_dir, self.site, 'fRootLitter_daily_perpft.nc'))
		# self.fStemLitter = xr.open_dataset(path.join(self.my_dir, self.site,'fStemLitter_daily_perpft.nc'))
		# self.actlyr = xr.open_dataset(path.join(self.my_dir, self.site,'actlyr_daily.nc'))
		# self.cRoot = xr.open_dataset(path.join(self.my_dir, self.site,'cRoot_daily_perpft.nc'))
		print('Input files loaded successfully')

		self.ignd = self.init.dims['layer'] # Number of soil layers
		self.ncell = 1  # Number of gridcells

		self.fLeafLitter = self.fLeafLitter['fLeafLitter'].data[:, :self.iccp2, 0, 0]      # Total carbon input from leaf to litter pool
		self.fLeafLitter = np.reshape(self.fLeafLitter, (self.ndays, self.iccp2))    # Reshaping and converting from s^-1 to yr^-1

		self.fRootLitter = self.fRootLitter['fRootLitter'].data[:, :self.iccp2, 0, 0]      # Total carbon input from Root to litter pool
		self.fRootLitter = np.reshape(self.fRootLitter, (self.ndays, self.iccp2))    # Reshaping and converting from s^-1 to yr^-1

		self.fStemLitter = self.fStemLitter['fStemLitter'].data[:, :self.iccp2, 0, 0]      # Total carbon input from stem to litter pool
		self.fStemLitter = np.reshape(self.fStemLitter, (self.ndays, self.iccp2))    # Reshaping and converting from s^-1 to yr^-1

		self.fcancmx = self.init['fcancmx'].data[0, :iccp2, 0, 0]                     # Fraction of pft coverage
		self.fcancmx = np.tile(self.fcancmx, (self.nruns, 1)).transpose()                  # Reshaping

		self.actlyr = self.actlyr.actlyr.data[:, :, 0]


		# Global variables
		# Reading and formatting variables from initialisation file
		self.SAND = self.init['SAND'].data[0, :, 0, 0]  # Sand content of soil [%]
		self.CLAY = self.init['CLAY'].data[0, :, 0, 0]  # Clay content of soil [%]

		# Reshaping SAND
		temp = np.zeros([self.ignd, self.ncell])
		temp[:, 0] = self.SAND
		self.SAND = temp

		# Reshaping CLAY
		temp = np.zeros([self.ignd, self.ncell])
		temp[:, 0] = self.CLAY
		self.CLAY = temp

		self.sdepth = self.init['SDEP'].data[0, 0, :]  # Soil Permeable depth [m]
		self.maxAnnualActLyr = self.init['maxAnnualActLyr'].data[0, 0, 0]  # Active layer thickness max over e-folding period

		# Initial state of carbon pools
		self.litrmass = self.init['litrmass'].data[0, :, :self.iccp2, 0, 0]  # Mass of C in litter pool [kg C m^-2]
		self.temp = np.zeros([self.ignd, self.iccp2, self.ncell])
		self.temp[:, :, 0] = self.litrmass
		self.litrmass = self.temp

		self.soilcmas = self.init['soilcmas'].data[0, :, :self.iccp2, 0, 0]  # Mass of C in soil pool [kg C m^-2]
		self.temp = np.zeros([self.ignd, self.iccp2, self.ncell])
		self.temp[:, :, 0] = self.soilcmas
		self.soilcmas = self.temp
		# Calculating delzw and zbotw
		self.isand = np.where((self.SAND == -4) | (self.SAND == -3))  # Flag for soil type -4 = ice, -3 = rock
		self.delz = self.init['DELZ'].data[:]  # Ground layer thickness [m]
		self.temp = np.zeros([self.ignd, self.ncell])
		self.temp[:, 0] = self.delz
		self.delz = self.temp
		self.zbot = np.zeros_like(self.delz)  # Dept of the bottom of each soil layer [m]
		for i in range(0, len(self.delz)):
			self.zbot[i] = np.sum(self.delz[:i + 1])

		self.delzw = np.zeros_like(self.delz)  # thickness of permeable part of soil layer
		self.isice = np.where(self.SAND == -4)  # Soil layers that are ice
		self.isrock = np.where(self.SAND == -3)  # Soil layers that are rock
		self.isbelow = np.where(self.sdepth >= self.zbot)  #
		self.isnear = np.where(self.sdepth < (self.zbot - self.delz + 0.025))

		if self.SAND[0] == -4:  # is ice
			self.delzw = self.delz
			self.SAND[:] = -4
		else:
			self.delzw = np.maximum(0.05, (self.sdepth - (self.zbot - self.delz)))

			self.delzw[self.isnear] = 0.0
			self.SAND[self.isnear] = -3

			self.delzw[self.isbelow] = self.delz[self.isbelow]
			self.delzw[self.isrock] = 0

		self.zbotw = np.maximum(0, (self.zbot - self.delz)) + self.delzw  # Bottom of soil layers [m]
		self.isand = np.where((self.SAND == -4) | (self.SAND == -3))  # Flag ,-4 = ice, -3 = ice


		# Soil properties variables
		self.mliq = self.mrsll['mrsll'].data[:, :, 0, 0]  # Mass of liquid water at each soil layer [kg m^-2]
		self.temp = np.zeros([self.ndays, self.ignd, 1])
		self.temp[:, :, 0] = self.mliq
		self.mliq = self.temp
		self.thliq = self.mliq / 1000 / self.delzw + self.moist_var # Conversion to m^3/m^3

		self.mice = self.mrsfl['mrsfl'].data[:, :, 0, 0]  # Mass of frozen water at each soil layer [kg m^-2]
		self.temp = np.zeros([self.ndays, self.ignd, 1])
		self.temp[:, :, 0] = self.mice
		self.mice = self.temp
		self.thice = self.mice / 1000 / self.delzw  # Conversion to m^3/m^3

		self.tbar = self.tsl['tsl'].data[:, :, 0, 0] + self.temp_var  # Temperature at each soil layer [K]
		self.temp = np.zeros([self.ndays, self.ignd, 1])
		self.temp[:, :, 0] = self.tbar
		self.tbar = self.temp
		self.thpor = (-0.126 * self.SAND + 48.9) / 100  # Soil total porority [m^3 m^-3]
		self.thpor[self.isand] = 0  # No value where soil is rock or ice
		self.b = 0.159 * self.CLAY + 2.91  # Clapp and Hornberger empirical param []
		self.b[self.isand] = 0  # No value where soil is rock or ice
		self.psisat = 0.01 * np.exp(-0.0302 *self. SAND + 4.33)  # Soil moisture suction at saturation [m]
		self.psisat[self.isand] = 0

		# Array of pft specific variables, 9 CTEM pfts + Bare + LUC
		self.apfts = np.array(
			[[0.4453, 0.5986, 0.6339, 0.7576, 0.6957, 0.6000, 0.6000, 0.5260, 0.5260, 0.5605, 0.5605],  # 0 bsratelt
			[0.0260, 0.0260, 0.0208, 0.0208, 0.0208, 0.0350, 0.0350, 0.0125, 0.0125, 0.02258, 0.02258],  # 1 bsratesc
			[0.42, 0.42, 0.53, 0.48, 0.48, 0.10, 0.10, 0.42, 0.42, 0.45, 0.45],  # 2 humicfac
			[4.70, 5.86, 3.87, 3.46, 3.97, 3.97, 3.97, 5.86, 4.92, 4.0, 4.0],  # 3 abar
			[1.85, 1.45, 2.45, 2.10, 2.10, 0.10, 0.10, 0.70, 0.70, 0.0, 0.0],  # 4 avertmas
			[0.80, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80, 0.80, 0.0, 0.0],  # 5 alpha
			[3.00, 3.00, 5.00, 5.00, 3.00, 2.00, 2.00, 1.00, 1.00, 0.0, 0.0]])  # 6 mxrtdpth

		self.abar = self.apfts[3, :]      # Parameter determining average root profile
		self.avertmas = self.apfts[4, :]  # average root biomass [kg C m^-2]
		self.alpha = self.apfts[5, :]
		self.mxrtdpth = self.apfts[6, :]

		# Computing carbon input from roots
		self.rmatctem = frmatctem.rmatctemf(self.zbotw, self.alpha, self.cRoot.cRoot.data[:, :11, 0, 0], self.avertmas, self.abar, self.sdepth, self.maxAnnualActLyr,
									self.mxrtdpth)
		self.root_input = (self.fRootLitter * self.rmatctem.transpose(2, 0, 1)).transpose(1, 2, 0)

		# self.itermax = 80000 * 2
		# self.itermax = 200

		self.delt = 1 / 365
		self.spinup = 100

	def run(self):
		"""
		Function that performs the simulation using the CLASSIC soil component. Simulates 1 or more specific sites
		:param apfts: pft specific parameter array
		:param nruns: number of different sites to simulate in a single run
		:param iccp2: number of pfts, 9 ctem pfts + bare + LUC
		:param self.ignd: number of soil layers
		:param litrmass: initial mass of carbon in the litter pool [kg C m^-2] (self.ignd, iccp1, nruns)
		:param soilcmas: initial mass of carbon in the soil pool [kg C m-2] (self.ignd, iccp1, nruns)
		:param self.ncell: number of gridcells to simulate
		:param tf: length of available data to drive model [yr]
		:param delt: length of one iteration [yr]
		:param ndays: number of days of available data [days]
		:param thice: Frozen water  content at each soil layer [m^3 m^-3] (ndays, self.ignd, nruns)
		:param thpor: Soil total porority [m^3 m^-3] (self.ignd, nruns)
		:param thliq: Liquid water content at each soil layer [m^3 m^-3] (ndays, self.ignd, nruns)
		:param psisat: Soil moisture suction at saturation [m] (self.ignd, nruns)
		:param isand: Soil content flag -4 = ice, -3 = bedrock
		:param tbar: Temperature at each soil layer [K] (ndays, self.ignd, nruns)
		:param b: Clapp and Hornberger empirical param [] (self.ignd, nruns)
		:param itermax: maximum number of iteration to simulate
		:param spinup: number of spinup iterations
		:param zbotw: Bottom of permeable soil layers [m] (self.ignd, nruns)
		:param actlyr: Active layer depth [m] (ndays, nruns)
		:param SAND: Sand content of soil [%] (self.ignd, nruns)
		:param fcancmx: pft coverage of gridcells [%] (iccp2, nruns)
		:param delzw: thickness of permeable part of soil layer [m] (self.ignd, nruns)
		:param root_input: Total Carbon Mass Flux from Roots to Litter [kg m^-2 s^-1]
		:param fLeafLitter: Total carbon mass flux from leaf to litter [kg m^-2 s^-1]
		:param fStemLitter: Total carbon mass flux from stem to litter [kg m^-2 s^-1]
		:return:
		"""
		
		# Initializing soil C content vector
		self.litrplsz = np.zeros([self.nruns, self.iccp2, self.ignd])
		self.soilplsz = np.zeros([self.nruns, self.iccp2, self.ignd])
		self.litrplsz += self.litrmass[:, :self.iccp2].reshape(self.iccp2, self.ignd)  # initial carbon mass of litter pool [kg C / m^2]
		self.soilplsz += self.soilcmas[:, :self.iccp2].reshape(self.iccp2, self.ignd)  # initial carbon mass of soil pool [kg C / m^2]

		self.litrplsz = self.litrplsz.transpose(1, 2, 0)
		self.soilplsz = self.soilplsz.transpose(1, 2, 0)

		# Initializing litter input array
		self.Cinput = np.zeros([self.iccp2, self.ignd, self.ncell])  # Pool carbon input

		# arrays for outputs
		self.litter = np.empty([self.ignd, self.nruns, self.itermax])
		self.soil = np.zeros([self.ignd, self.nruns, self.itermax])
		self.resp = np.zeros([self.ignd, self.iccp2, self.nruns, self.itermax])

		count = 0  # iteration count
		self.bsratelt = self.apfts[0, :]  # turnover rate of litter pool [kg C/kg C.yr]
		self.bsratesc = self.apfts[1, :]  # turnover rate of soil pool [kg C/kg C.yr]
		self.humicfac = self.apfts[2, :]  # Humification factor
		self.kterm = 3  # constant that determines the depth at which turbation stops
		self.biodiffus = 2.73972e-07  # Diffusivity coefficient for biodiffus [m^2 / day]
		self.cryodiffus = 1.36986e-06  # Diffusivity coefficient for cryodiffus [m^2 / day]
		self.r_depthredu = 10  # depth reduction parameter


		self.reduceatdepth = np.exp(-self.delzw / self.r_depthredu)  # Depth reduction factor

		self.sample = int(4 / self.delt)
		# Computing env. modifiers beforehand
		self.envmod = fenv.fenv(self.ndays, self.thice, self.thpor, self.thliq, self.psisat, self.ignd, self.isand, self.tbar, self.b, self.nruns)
		print('Envmod successfully calculated')
		self.envmodltr = self.envmod[0]
		self.envmodsol = self.envmod[1]
		# Time loop
		print('Starting model run')
		with alive_bar(self.itermax, bar='smooth', spinner='waves2') as bar:  # Setting up progress bar
			while count < self.itermax:

				t = count % self.sample  # cycling on the length of imput files
				# Computing environmental modifier at current timestep
				self.envltr = self.envmodltr[t]  # Litter pool environmental modifier at current timestep
				self.envsol = self.envmodsol[t]  # Soil pool environmental modifier at current timestep

				# Modifying transfert coeff if spinup=True
				if count <= self.spinup:
					self.spinfast = 10  # transfer coeff from litter pool to soil
				else:
					self.spinfast = 1

				# Computing input vector at current time step
				self.Cinput[:, 0, 0] = (self.fLeafLitter[t] + self.fStemLitter[t])  # Carbon input to litter pool at timestep t [Kg C m^-2 yr^-1]
				#Cinput[:, :, 0] += root_input[t]  # Carbon input from roots at timestep t [Kg C m^-2 yr^-1]

				# Computing respiration over time step for both pool
				self.ltresveg = self.bsratelt * (self.litrplsz * self.envltr * self.reduceatdepth).transpose(1, 2, 0)  # [kg C m^-2 yr^-1]
				self.scresveg = self.bsratesc * (self.soilplsz * self.envsol * self.reduceatdepth).transpose(1, 2, 0)  # [kg C m^-2 yr^-1]

				# Computing pool size change in both pool
				self.dx1 = self.Cinput - (self.ltresveg * (1 + self.humicfac)).transpose(2, 0, 1)
				self.dx2 = self.spinfast * ((self.ltresveg * self.humicfac) - self.scresveg).transpose(2, 0, 1)

				# Updating pools
				self.litrplsz += self.dx1 * self.delt  # Updating litter pool [kg C m^-2]
				self.soilplsz += self.dx2 * self.delt  # Updating soil pool [kg C m^-2]

				# Calling turbation subroutine
				self.litrplsz, self.soilplsz = turbationf.turbation(
					self.litrplsz, 
					self.soilplsz,
					self.zbotw, 
					self.ignd, 
					self.kterm, 
					self.actlyr[t], 
					self.cryodiffus,
					self.biodiffus, 
					self.spinfast, 
					self.SAND, 
					self.nruns, 
					self.delt, 
					self.iccp2)

				self.gridltr = np.sum(self.litrplsz.transpose(1, 0, 2) * self.fcancmx, axis=1)  # Litter pool size at grid level
				self.gridsol = np.sum(self.soilplsz.transpose(1, 0, 2) * self.fcancmx, axis=1)  # SoilC pool size at grid level
				self.gridrsp = np.sum((self.ltresveg + self.scresveg) * self.fcancmx, axis=1)  # Respiration at grid level
				# Adding to output arrays
				self.litter[:, :, count] = self.gridltr
				self.soil[:, :, count] = self.gridsol
				self.resp[:, :, 0, count] = self.gridrsp

				count += 1  # Iteration count
				bar()  # Update progress bar
		self.litr_output,self.soil_output,self.resp_output = self.litter, self.soil, self.resp
		print(f"Finished test on site {self.site} with temp {self.temp_var} at moisture {self.moist_var}")
		
	def test(test: test.Test):
		model = Model(
			test.temp_var, 
			test.moist_var,
			test.site, 
			test.nruns, 
			test.iccp2, 
			test.dir,
			itermax = test.maxiters
		)
		model.run()
		model.save(test.fp)
		model.plot(test.fp,test.name)
		return model.resp_output
		# self.litr_output, self.soil_output, self.resp_output = fmodel.model(apfts, nruns, iccp2, self.ignd, litrmass, soilcmas, self.ncell, tf, delt, ndays,
		# 											thice, thpor, thliq, psisat, isand, tbar, b, itermax, spinup, zbotw,
		# 											actlyr.actlyr.data[:, :, 0], SAND, fcancmx, delzw, root_input,
		# 											fLeafLitter, fStemLitter)
		# fplot.fplot(litr_output,soil_output,resp_output,site)
		
		# return litr_output, soil_output, resp_output
	def save(self, fp=''):
		if (not path.isdir(path.join('test_data',self.site))):
				mkdir(path.join('test_data',self.site))
		df = pd.DataFrame.from_dict({
			'litr': np.sum(self.litr_output,axis=(0,1)),
			'soil': np.sum(self.soil_output,axis=(0,1))
		})
		df.index.name = 'iter'
		df.to_csv(f"test_data/{self.site}/{fp}.csv")
	def plot(self,fp='', title=''):
		'''
		Function that plots model outputs
		:param output: Model output, no formatting needed
		:return:
		'''
		x = np.arange(0,self.litr_output.shape[2])

		# Formating outputs
		litr_format = np.sum(self.litr_output,axis=(0,1))
		soil_format = np.sum(self.soil_output,axis=(0,1))
		resp_format = round(np.mean(np.sum(self.resp_output, axis=(0,1,2))), 3)

		# plotting results
		plt.cla()
		plt.plot(x, litr_format, '-b', label='Litter pool')
		plt.plot(x, soil_format, '-r', label='Soil pool')
		plt.axhline(resp_format, ls=':', c='g')
		plt.text(20000,10,'Average repiration:'+ str(resp_format))

		# Formatting plot
		plt.title(f"Soil carbon simulation of fluxnet site {self.site} with a {self.temp_var} temperature increase")
		plt.ylabel('Total carbon content [kg C $m^{-2}$]')
		plt.xlabel('Number of days simulated')
		plt.legend()
		if fp:
			if (not path.isdir(path.join('test_graphs',self.site))):
				mkdir(path.join('test_graphs',self.site))
			plt.savefig(f"test_graphs/{self.site}/{fp}.png")
			plt.rc('pgf', texsystem='xelatex')
			plt.savefig(f"test_graphs/{self.site}/{fp}.pgf")
		else:
			plt.show()