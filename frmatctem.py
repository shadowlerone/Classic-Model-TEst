# rmatctem function script
import numpy as np

def rmatctemf(zbotw, alpha, rootmass, avertmas, abar, soildpth, maxannualactlyr, mxrtdpth):
    """
    Function that calculates the variable rmatctem, the fraction of live roots per soil layer per pft
    :param zbotw: Bottom of soil layers [m] dim(ignd)
    :param alpha: Parameter determining how roots grow dim(icc)
    :param rootmass: Root mass [kg C m^-2] dim(ndays,icc)
    :param avertmas: Average root biomass [kg C m^-2] dim(icc)
    :param abar: Parameter determining average root profile dim(icc)
    :param soildpth: Soil permeable depth [m] dim(scalar)
    :param maxannualactlyr: Active layer thickness maximum over the e-folding period specified by parameter eftime [m] dim(scalar)
    :param mxrtdpth: Maximum rooting depth [m] dim(icc)
    :return: rmatctem, fraction of live roots in each soil layer for each pft dim(icc, ignd)
    """

    ndays = rootmass.shape[0]
    # Estimating parameters b and alpha
    b = abar * (avertmas ** alpha)
    useb = b                                        # per pft
    usealpha = alpha                                # per pft
    rootdpth = (4.605 * (rootmass ** alpha)) / b    # Depth of roots [m]
    a = np.zeros([ndays,11])

    # Defining conditions on rootdpth
    i1 = np.where(rootdpth > np.minimum(np.minimum(mxrtdpth, soildpth), maxannualactlyr))
    i2 = np.where(rootdpth <= np.minimum(np.minimum(mxrtdpth, soildpth), maxannualactlyr))

    # Applying conditions
    rootdpth[i1] = np.minimum(np.minimum(mxrtdpth[i1[1]], soildpth), maxannualactlyr)
    a[i1] = 4.605 / rootdpth[i1]
    a[i2] = useb[i2[1]] / (rootmass[i2] ** usealpha[i2[1]])

    # If rootdpth = 0 then 100% of roots are on top layer
    i3 = np.where(rootdpth <= 1e-12)
    a[i3] = 100.0

    # Computing rmatctemp
    kend = np.zeros([ndays,11]) + 9999                      # soil layer in which the roots end, initialized at a dummy value
    totala = 1.0 - np.exp(- a * rootdpth)
    dzbotw = np.tile(zbotw, (11,1,ndays)).transpose(2,0,1)   # Modifying shape for operations on array
    dzroot = np.tile(rootdpth,(20,1,1)).transpose(1,2,0)     # Modifying shape for operations on array

    # Finding in which soil layer the roots end
    ishallow = np.where(rootdpth <= zbotw[0])          # Roots end before or in first soil layer
    iinbetween = np.where((dzroot <= dzbotw) & (dzroot > np.roll(dzbotw, 1, axis=2)))  # roots are at this soil layer
    kend[ishallow] = 0                              # Layer 0
    kend[iinbetween[0],iinbetween[1]] = iinbetween[2]             # Layer at which the roots end (per pft)

    # Computing rmatctem
    ipft = np.arange(0,len(kend),1)
    i = np.where(kend < 1e6)
    i = (i[0],i[1],kend.flatten().astype(int))
    ii = kend.flatten()
    # applying conditions
    etmp = np.exp(-(a * dzbotw.transpose(2,0,1))).transpose(1,2,0)             # in general
    rmatctema = (np.roll(etmp,1,axis=2) - etmp) / np.tile(totala, (20,1,1)).transpose(1,2,0)  # in general
    etmp[:,:,0] = np.exp(-a*dzbotw[:,:,0])               # etmp at first layer
    rmatctema[:,:,0] = (1 - etmp[:,:,0] / totala)       # rmatctem at first layer

    # Looping on every end layer and computing rmat value there
    for k in range(0,kend.shape[0]):
        for h in range(0,kend.shape[1]):
            etmp[k,h,kend[k,h].astype(int)] = -np.exp(-a[k,h]*rootdpth[k,h])
            rmatctema[k,h,kend[k,h].astype(int)] = (etmp[k,h,kend[k,h].astype(int) - 1] - etmp[k,h,kend[k,h].astype(int)]) / totala[k,h]

            rmatctema[k,h,kend[k,h].astype(int)+1:] = 0 # Setting rmatctema to zeros when deeper than kend

    rmatctema[ishallow[0],ishallow[1],0] = 1                      # for pft where roots stop at first layer, 100% in first
    rmatctema[ishallow[0],ishallow[1],1:] = 0                     # 0% in all other layers

    return rmatctema