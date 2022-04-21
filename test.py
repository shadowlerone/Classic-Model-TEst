class Test:
	def __init__(
		self, name, directory, site, fp, temp_var, moist_var, nruns=1, iccp2 = 11, tf = 19, maxiters=80000*2):
		self.dir = directory
		self.site = site
		self.fp = fp
		self.temp_var = temp_var
		self.moist_var = moist_var
		self.nruns = nruns
		self.iccp2 = iccp2
		self.tf = tf
		self.name = name
		self.maxiters=maxiters