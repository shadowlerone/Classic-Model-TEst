# Turbation function that takes current C pools and perform turbation on them
import numpy as np

def tridiag_mat_alg(a, b, c, r, u):
    """
    Performs the tridiagonal matrix algorithm, see https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    :param a: subdiagonal a
    :param b: diagonal b
    :param c: super diagonal c
    :param r: right-hand side of Crank-Nicolson solver
    :param u: vector of soil layers carbon
    :return: updated vector
    """
    n = len(b)
    bet = b[0]
    u[0] = r[0] / bet
    gam = np.zeros(len(b))

    # decomposition and foward substitution
    for i in range(1,n):
        gam[i] = c[i-1] / bet
        bet = b[i] - a[i] * gam[i]
        u[i] = (r[i] - a[i] * u[i-1]) / bet
    # Back substitution
    for j in range(n-2,0,-1):
        u[j] = u[j] - gam[j+1] * u[j+1]
    return u

def turbation(litrmass, soilcmass, zbotw, ignd,kterm, actlyr,cryodiffus,biodiffus,spinfast,SAND,ncell,delt,iccp2):
    """
    Function that performs turbation on litter and soil pools using the Crank-Nicolson method, described at the link
    below: https://en.wikipedia.org/wiki/Crank%E2%80%93Nicolson_method
    :param litrmass: Mass of carbon per layer, per pft [kg C m^-2] in the litter pool
    :param soilcmass: Mass of carbon per layer, per pft [kg C m^-2] in the soil pool
    :param zbotw: Bottom of soil layer [m]
    :param ignd: Number of soil layers
    :param kterm: Constant used in determination of depth at which cryoturbation ceases
    :param actlyr: Active layer depth [m], for all gridcells at current timestep
    :param cryodiffus: Cryoturbation diffusion coefficient [m^2 day^-1]
    :param biodiffus: Bioturbation diffusion coefficient [m^2 day^-1]
    :param spinfast: Spin up coefficient
    :param SAND: Sand content of soil [%]
    :param ncell: Number of gridcells
    :param delt: length of timestep [yr]
    :param iccp2: Number of PFTs
    :return: updated litrmass and soilsmass
    """
    # Finding bottom layer of the permeable soil
    botlyr = (ignd - 1) - np.sum((SAND == -3 | -4), axis=0)

    # Where there is permeable soil, add two layers for boundary condition
    iperm = np.where(botlyr > 0)
    botlyr[iperm] += 2

    # Store the size of present pools for later comparison
    psoilc = np.sum(soilcmass, axis=1)  # Sum on soil layers
    plit = np.sum(litrmass, axis=1)     # Sum on soil layers
    for i in range(0,ncell):            # Looping on every gridcell
        for j in range(0,iccp2-1):      # Looping on every pft except LUC
            if (psoilc[j,i] > 1e-13):   # If there is carbon, perform turbation
                # Creating empty arrays for tridiagonal solver
                soilcinter = np.zeros(botlyr[i])
                littinter = np.zeros(botlyr[i])
                depthinter = np.zeros(botlyr[i])

                # Applying boundary condition at surface interface
                soilcinter[0] = 0
                littinter[0] = 0
                depthinter[0] = 0

                # Put soil/litter C in arrays, index 0 is surface and botlyr is bedrock bottom
                soilcinter[1:botlyr[i]-1] = soilcmass[j,0:botlyr[i]-2,i]
                littinter[1:botlyr[i]-1] = litrmass[j,0:botlyr[i]-2,i]
                depthinter[1:botlyr[i]-1] = zbotw[0:botlyr[i] - 2,i]

                # Boundary condition at bedrock interface
                soilcinter[-1] = 0
                littinter[-1] = 0

                # Check for special case where soil = permeable all the way to bottom
                if botlyr[i] <= ignd-1:
                    botthick = zbotw[botlyr[i],i]
                else:
                    botthick = zbotw[-1,i]
                depthinter[-1] = botthick

                # Determining the effective diffusion coefficients for each soil layers
                kactlyr = actlyr[i] * kterm
                if actlyr[i] <= 1.: # cryoturb only if actlyr is shallower than 1m
                    diffus = cryodiffus
                else: # Bioturb is dominant
                    diffus = biodiffus
                ishallow = np.where(depthinter < actlyr[i]) # Shallow, so vigorous cryoturbation
                ilinear = np.where((depthinter > actlyr[i]) & (depthinter < kactlyr))
                effectiveD = np.zeros(botlyr[i])
                effectiveD[ishallow] = diffus * spinfast
                effectiveD[ilinear] = diffus * (1-(depthinter[ilinear] - actlyr[i])/((kterm -1) * actlyr[i])) * spinfast

                # Set up coeff for tridiagonal matrix solver
                avect = np.zeros(botlyr[i])
                bvect = np.zeros(botlyr[i])
                cvect = np.zeros(botlyr[i])
                rvect_sc = np.zeros(botlyr[i])
                rvect_lt = np.zeros(botlyr[i])

                # Applying the Crank-Nicolson method
                # Upper boundary condition
                avect[0] = 0
                bvect[0] = 1
                cvect[0] = 0
                rvect_sc[0] = 0
                rvect_lt[0] = 0

                # Soil layers
                dzm = depthinter - np.roll(depthinter, 1)
                termr = effectiveD * delt / dzm**2
                avect = -termr
                bvect = 2 * (1+termr)
                cvect = -termr
                rvect_sc = termr * np.roll(soilcinter, 1) + 2*(1 - termr) * soilcinter + termr * np.roll(soilcinter, -1)
                rvect_lt = termr * np.roll(littinter, 1) + 2*(1-termr) * littinter + termr * np.roll(littinter, -1)

                # Bottom boundary condition
                avect[botlyr[i]-1] = 0.0
                bvect[botlyr[i]-1] = 1.0
                cvect[botlyr[i]-1] = 0.0
                rvect_sc[botlyr[i]-1] = 0.0
                rvect_lt[botlyr[i]-1] = 0.0

                # Using the tridiagonal solver to get the new C mass
                soilcinter = tridiag_mat_alg(avect,bvect,cvect,rvect_sc,soilcinter)     # Soil pool
                littinter = tridiag_mat_alg(avect,bvect,cvect,rvect_lt,littinter)       # Litter pool

                # Putting C back into the original arrays
                soilcmass[j,0:botlyr[i]-2,i] = soilcinter[1:botlyr[i]-1]
                litrmass[j,0:botlyr[i]-2,i] = littinter[1:botlyr[i]-1]

    # Computing Carbon mass after turbation for mass conservation check
    asoilc = np.sum(soilcmass, axis=1)
    alit = np.sum(litrmass, axis=1)

    # Mass conservation check
    amounts = psoilc - asoilc
    amountl = plit - alit

    # If C was gained/lost we had it back to all layers
    for i in range(0,ignd):
        soilcmass[:,i,:] += amounts/(botlyr-2)
        litrmass[:,i,:] += amountl/(botlyr-2)

    return litrmass, soilcmass