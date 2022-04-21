import numpy as np
from cachetools import cached, LRUCache, TTLCache

# @cached(cache=TTLCache(maxsize=1024, ttl=600))
def fenv(ndays, thice, thpor, thliq, psisat, ignd,isand,tbar,b,ncell):
    """
    Function that computes environmental scalars
    :param ndays: Number of days of available data
    :param thice: Frozen water  content at each soil layer [m^3 m^-3] (ndays, ignd, nruns)
    :param thpor: Soil total porority [m^3 m^-3] (ignd, nruns)
    :param thliq: Liquid water content at each soil layer [m^3 m^-3] (ndays, ignd, nruns)
    :param psisat: Soil moisture suction at saturation [m] (ignd, nruns)
    :param ignd: Number of soil layers
    :param isand: Soil content flag -4 = ice, -3 = bedrock
    :param tbar: Temperature at each soil layer [K] (ndays, ignd, nruns)
    :param b: Clapp and Hornberger empirical param [] (ignd, nruns)
    :param ncell: Number of gridcells
    :return: environmental modifier
    """
    # Defining variables
    psisat = np.tile(psisat, (ndays,1,1))
    tcrit = -1.0  # Temperature below which respiration is inhibited [C]
    tfrez = 273.16  # Freezing point of water [K]
    frozered = 0.1  # Factor to reduce resp. by for temp below tcrit

    # Find the matric potential
    ipore = np.where(thice > thpor)  # Where the whole pore space is ice
    psi = psisat * (thliq / (thpor - thice))**(-b)
    psi[ipore] = 10000.0    # pore space is ice, suction is very high

    # Find moisture scalar
    ltrmoscl = np.zeros([ndays, ignd, ncell])  # Litter pool moisture scalar
    socmoscl = np.zeros([ndays, ignd, ncell])  # Soil pool moisture scalar

    # Different conditions on psi
    cvrdry = np.where(psi >= 10000.0)    # Soil is very dry
    cdrmst = np.where((6 < psi) & (psi < 10000.0)) # Soil is dry to moist
    coptim = np.where((4 <= psi) & (psi <= 6))  # Soil is optimal
    ctoowt = np.where((psisat < psi) & (psi < 4)) # Soil is too wet
    csatur = np.where(psi <= psisat) # Soil is saturated

    # Computing the moisture scalars
    ltrmoscl[cvrdry] = 0.2
    socmoscl[cvrdry] = 0.2

    ltrmoscl[cdrmst] = 1.0 - 0.8 * ((np.log10(psi[cdrmst]) - np.log10(6.0)) / (np.log10(10000.0) - np.log10(6.0)))
    socmoscl[cdrmst] = ltrmoscl[cdrmst]

    ltrmoscl[coptim] = 1
    socmoscl[coptim] = 1

    socmoscl[ctoowt] = 1.0 - 0.5 * ((np.log10(4.0) - np.log10(psi[ctoowt])) / (np.log10(4.0) - np.log10(psisat[ctoowt])))
    ltrmoscl[ctoowt] = socmoscl[ctoowt]
    if 0 in np.unique(ctoowt):
        ltrmoscl[0] = 1

    socmoscl[csatur] = 0.5
    ltrmoscl[csatur] = socmoscl[csatur]
    if 0 in np.unique(csatur):
        ltrmoscl[0] = 1
    for i in range(0,ndays):
        ltrmoscl[i][isand] = 0.2
        socmoscl[i][isand] = 0.2

    ltrmoscl.clip(0.2, 1.0, out=ltrmoscl)
    socmoscl.clip(0.2, 1.0, out=socmoscl)

    ff_p = np.array([ltrmoscl, socmoscl])

    # Computing q10 response function
    litrq10 = 1.44 + 0.56 * np.tanh(0.075 * (46.0 - (tbar - tfrez)))
    solcq10 = litrq10

    q10funcLitr = litrq10**(0.1 * (tbar - tfrez - 15.0))
    q10funcSoilC= solcq10**(0.1 * (tbar - tfrez - 15.0))

    # Accounting for frozen soil
    ifroz = np.where((tbar - tfrez) <= tcrit)
    q10funcLitr[ifroz] *= frozered
    q10funcSoilC[ifroz] *= frozered
    ff_T = np.array([q10funcLitr,q10funcSoilC])
    envmod = ff_T * ff_p
    return envmod