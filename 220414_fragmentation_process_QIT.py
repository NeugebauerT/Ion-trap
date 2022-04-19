""" This Python script performs the calculations described in the scientific article
"Calculating the fragmentation process in quadrupole ion traps", T.S. Neugebauer
and T. Drewello, accepted in the International Journal of Mass Spectrometry
in February 2022.    
"""

import sys
import matplotlib.pyplot as plt
import os
import math
from datetime import datetime
import numpy as np

print(sys.version)

# important calculation parameters that affect the calculation accuracy and calculation time
harmonic_limit = 5                      # calculating expansion coefficients up to +-n. Affects calculational time. but not linearly. Default value: 5
phi_steps_phasespace = 900              # number of steps of 2phi = 180° RF-phases for the velocity calculations. Hardly affects calculation time. Default value: 900
number_mass_segments = 50              # number of datapoints per plot. Linearly affects calculation time. Default value: 50
phi_steps = 90                          # number of steps of 2phi = 180° RF-phases for the full calculation. Linearly affects calculation time. Default value: 90
delta_prec_steps = 45                   # Number of steps, the precursor's underlying harmonic motion is split into per RF-phase. Linearly affects calculation time. Default value: 45

LMCO_to_calculate = [16, 27, 40]                # The Lowmass cutoff values in % calculated. Several values linearly affects calculational time. Default fragmentation LMCO 10 to 40.
precursormass_to_calculate = [2000]     # only important if Coulomb repulsion is included. Several values linearly affect calculation time. Default value: 2000 m/z
coulomb_factor_segments = 0             # number of steps between 0 and 100% Coulomb repulsion used. Linearly affects calculation time. Default value: 0 (no Coulomb repulsion)


def frequency(q: float, a: float = 0) -> float:
    """
    This function calculates the fundamental frequency Beta of the underlying harmonic motion
    in a parametric oscillator, i.e. an oscillator that is described by Mathieu's equation.
    :param q: Mathieu stability parameter q
    :param a: Mathieu stability parameter a
    :return: returns the fundamental oscillation frequency Beta
    """
    beta_test = 0.5
    iteration = 0
    while True:
        beta2 = (a + (q**2 / ((beta_test + 2)**2 - a
                      - (q**2 / ((beta_test + 4)**2 - a
                       - (q**2 / ((beta_test + 6)**2 - a
                        - (q**2 / ((beta_test + 8)**2 - a
                         - (q**2 / ((beta_test + 10)**2 - a))))))))))
                   + (q**2 / ((beta_test - 2)**2 - a
                      - (q**2 / ((beta_test - 4)**2 - a
                       - (q**2 / ((beta_test - 6)**2 - a
                        - (q**2 / ((beta_test - 8)**2 - a
                         - (q**2 / ((beta_test - 10)**2 - a))))))))))
                 )**0.5
        iteration += 1
        if abs((beta2 / beta_test - 1)) < 10**(-14) or beta2 == beta_test:
            return beta2
        beta_test = beta2
        if iteration > 1000:
            print("*" * 80 + "\n" + "*" * 80 + "\n" + "*" * 80 + "\n")
            print("Frequency did not converge correctly.")
            print("*" * 80 + "\n" + "*" * 80 + "\n" + "*" * 80 + "\n")
            return beta2


# trap dimensions
r0 = 7.25 * 2**0.5  # in mm
z0 = 7.25  # in mm
fRF = 781  # in kHz

# scientific constants
AVOGADRO = 6.02214086 * 10**26  # in mol / kg
ELECTRON = 1.602176634 * 10**(-19)  # in As
VACUUM_PERMITTIVITY = 8.8541878128 * 10**(-12)  # in As / Vm
BOLTZMANN_CONSTANT = 1.380649 * 10**(-23)  # in kgm^2 / s^2K
PI = np.pi

# constants for better readability
C2N = 0
C2N_TIMES_FREQ = 1
C2N_TIMES_FREQSQUARED = 2
PREC = "prec"
FRAG = "frag"
LOST = 0
TRAPPED = 1
COLD = 2
MAX = 3
MIN = 4
SUM = 0
COUNT = 1
VALUE = 0
PHI = 1
DELTA = 2
# instrument information
temp_instrument_C = 40.0  # in °C
temp_instrument = temp_instrument_C + 273.15  # in K
mass_gas_u = 4.0026  # in u
mass_gas = mass_gas_u / AVOGADRO  # in kg
T_eff_prec_fragmentation = 700  # in K

plt.rcParams.update({'font.family': 'Cambria'})
plt.rcParams["mathtext.fontset"] = "cm"

print("#" * 80)
print(f"The used trap dimensions are: r0 = {r0:.3f} mm, z0 = {z0:.3f} mm and fRF = {fRF:.1f} kHz")
print(f"Instrument temperature = {temp_instrument_C:.1f} °C or {temp_instrument:.1f} K")
print(f"Mass collision gas = {mass_gas_u}u or {mass_gas} kg")
print(f"Effective collisional temperature needed for precursor fragmentation = {T_eff_prec_fragmentation} K "
      f"or {T_eff_prec_fragmentation - 273.15} °C")
print("#" * 80)

# dictionaries into which results will be written
summed_Coulomb_factors = {}
fragmentation_factors = {}


# start of calculation
for index_cutoff, cutoff in enumerate(LMCO_to_calculate):

    # creating result dictionary structure
    fragmentation_factors[cutoff] = {}
    summed_Coulomb_factors[cutoff] = {}

    LMCO = cutoff  # in %
    qLMCO = 0.90804633
    q_prec = LMCO / 100 * 0.90804633
    a_prec = 0

    # calculating secular frequency of precursor beta_prec
    beta_prec = frequency(q=q_prec, a=a_prec)

    print("#" * 80)
    print(f"Fragmentation low-mass cutoff = {LMCO:.1f}%: q = {q_prec:.6f}, a = {a_prec:.6f} beta = {beta_prec:.6f}")

    exp_coeff = {"prec": {}, "frag": {}}
    sum_exp_coeff = {"prec": [], "frag": []}
    exp_coeff[PREC][0] = (1.0, 1 * (beta_prec + 2 * 0), 1 * (beta_prec + 2 * 0)**2)
    # same as: exp_coeff[PREC].update({0: (1.0, 1 * (beta_prec + 2 * 0), 1 * (beta_prec + 2 * 0)**2)})
    sum_exp_coeff[PREC] = [1.0, beta_prec, beta_prec**2]
    for n in range(harmonic_limit + 1):
        C = exp_coeff[PREC][n][0] * (- q_prec) / ((beta_prec + 2 * (n + 1))**2 - a_prec - (q_prec**2
                                                / ((beta_prec + 2 * (n + 2))**2 - a_prec - (q_prec**2
                                                 / ((beta_prec + 2 * (n + 3))**2 - a_prec - (q_prec**2
                                                  / ((beta_prec + 2 * (n + 4))**2 - a_prec - (q_prec**2
                                                   / ((beta_prec + 2 * (n + 5))**2 - a_prec)))))))))
        exp_coeff[PREC][n + 1] = (C, C * (beta_prec + 2 * (n + 1)), C * (beta_prec + 2 * (n + 1))**2)
        sum_exp_coeff[PREC][C2N] += exp_coeff[PREC][n + 1][C2N]
        sum_exp_coeff[PREC][C2N_TIMES_FREQ] += exp_coeff[PREC][n + 1][C2N_TIMES_FREQ]
        sum_exp_coeff[PREC][C2N_TIMES_FREQSQUARED] += exp_coeff[PREC][n + 1][C2N_TIMES_FREQSQUARED]
    for n in range(0, -(harmonic_limit + 1), -1):
        C = exp_coeff[PREC][n][0] * (- q_prec) / ((beta_prec + 2 * (n - 1))**2 - a_prec - (q_prec**2
                                                / ((beta_prec + 2 * (n - 2))**2 - a_prec - (q_prec**2
                                                 / ((beta_prec + 2 * (n - 3))**2 - a_prec - (q_prec**2
                                                  / ((beta_prec + 2 * (n - 4))**2 - a_prec - (q_prec**2
                                                   / ((beta_prec + 2 * (n - 5))**2 - a_prec)))))))))
        exp_coeff[PREC][n - 1] = (C, C * (beta_prec + 2 * (n - 1)), C * (beta_prec + 2 * (n - 1))**2)
        sum_exp_coeff[PREC][C2N] += exp_coeff[PREC][n - 1][C2N]
        sum_exp_coeff[PREC][C2N_TIMES_FREQ] += exp_coeff[PREC][n - 1][C2N_TIMES_FREQ]
        sum_exp_coeff[PREC][C2N_TIMES_FREQSQUARED] += exp_coeff[PREC][n - 1][C2N_TIMES_FREQSQUARED]
    exp_coeff_prec_sorted = sorted(list(exp_coeff[PREC].items()))

    # calculating precursor maximum displacement umax, and average position and velocities
    aPosition_avg_prec = 0
    aPosition_squared_avg_prec = 0
    aVelocity_avg_prec = 0
    aVelocity_squared_avg_prec = 0
    aPosition_max_prec = 0
    aVelocity_max_prec = 0
    phi_segments = 0
    for phi in np.linspace(0, 90, phi_steps_phasespace + 1):

        # using my approach for calculation: https://doi.org/10.1016/j.ijms.2021.116641
        aPosition_prec = 0.0
        aPosition_squared_prec = 0.0
        aVelocity_prec = 0.0
        aVelocity_squared_prec = 0.0
        for m in range(-harmonic_limit, harmonic_limit + 1, 1):
            for o in range(-harmonic_limit, harmonic_limit + 1, 1):
                aPosition_squared_prec += exp_coeff[PREC][m][C2N] * exp_coeff[PREC][o][C2N]\
                                          * np.cos((m - o) * 2*phi / 360 * 2*PI)
                aVelocity_squared_prec += exp_coeff[PREC][m][C2N_TIMES_FREQ] * exp_coeff[PREC][o][C2N_TIMES_FREQ]\
                                          * np.cos((m - o) * 2*phi / 360 * 2*PI)
        aPosition_prec = aPosition_squared_prec**0.5
        aVelocity_prec = aVelocity_squared_prec**0.5
        if aPosition_prec > aPosition_max_prec:
            aPosition_max_prec = aPosition_prec
            phi_aPosition_max_prec = phi
        if aVelocity_prec > aVelocity_max_prec:
            aVelocity_max_prec = aVelocity_prec
            phi_aVelocity_max_prec = phi
        if phi == 0 or phi == 90:
            aPosition_avg_prec += aPosition_prec
            aPosition_squared_avg_prec += aPosition_squared_prec
            aVelocity_avg_prec += aVelocity_prec
            aVelocity_squared_avg_prec += aVelocity_squared_prec
            phi_segments += 1
        else:
            aPosition_avg_prec += 2 * aPosition_prec
            aPosition_squared_avg_prec += 2 * aPosition_squared_prec
            aVelocity_avg_prec += 2 * aVelocity_prec
            aVelocity_squared_avg_prec += 2 * aVelocity_squared_prec
            phi_segments += 2
    umax_prec = aPosition_max_prec
    upmax_prec = aVelocity_max_prec
    zpmax_prec = upmax_prec / umax_prec * z0 / 1000 * 2*PI * fRF * 1000 / 2
    uavg_prec = aPosition_avg_prec / phi_segments * 2 / PI
    upavg_prec = aVelocity_avg_prec / phi_segments * 2 / PI
    zpavg_prec = upavg_prec / umax_prec * z0 / 1000 * 2*PI * fRF * 1000 / 2
    uRMS_prec = (aPosition_squared_avg_prec / phi_segments)**0.5 / 2**0.5
    upRMS_prec = (aVelocity_squared_avg_prec / phi_segments)**0.5 / 2**0.5
    zpRMS_prec = upRMS_prec / umax_prec * z0 / 1000 * 2*PI * fRF * 1000 / 2

    for index_precmass, precursormass in enumerate(precursormass_to_calculate):

        # creating result dictionary structure
        fragmentation_factors[cutoff][precursormass] = {}
        summed_Coulomb_factors[cutoff][precursormass] = {}

        mass_prec_u = precursormass  # in u
        mass_prec = mass_prec_u / AVOGADRO  # in kg
        charge_prec_z = 2  # in e
        charge_prec = charge_prec_z * ELECTRON  # in As
        reduced_mass_u = mass_prec_u * mass_gas_u / (mass_prec_u + mass_gas_u)  # in u
        reduced_mass = reduced_mass_u / AVOGADRO  # in kg
        zpRMS_prec_fragmentation = ((T_eff_prec_fragmentation - reduced_mass / mass_gas * temp_instrument)
                                    * 3 * BOLTZMANN_CONSTANT / reduced_mass)**0.5
        zmax_prec_fragmentation = zpRMS_prec_fragmentation * umax_prec / upRMS_prec * 1000 / (2*PI * fRF * 1000 / 2)
        charge_distance_prec_A = 10.0 * (mass_prec_u / 2000)**(1 / 3.0)  # in Angström
        charge_distace_prec = charge_distance_prec_A / 10**10  # in m
        Epot_Coulomb = 1 / (4 * PI * VACUUM_PERMITTIVITY) * ELECTRON**2 / charge_distace_prec / ELECTRON  # in eV

        print("#" * 80)
        print(f"Precursor mass: {mass_prec_u}u or {mass_prec} kg")
        print(f"Precursor charge z: {charge_prec_z} or {charge_prec} As")
        print(f"Reduced mass (with collision gas) = {reduced_mass_u}u or {reduced_mass} kg")
        print(f"The precursor ion needs zpRMS = {zpRMS_prec_fragmentation:.1f} m/s to reach the required "
              f"{T_eff_prec_fragmentation:.1f} K effective collisional temperature to fragment")
        print(f"This corresponds to an oscillation amplitude of {zmax_prec_fragmentation:.3f} mm for the used "
              f"precursor q-value of {q_prec}")
        print(f"Charge distance in precursor ion: {charge_distance_prec_A} A or {charge_distace_prec} m")
        print(f"Coulomb potential: {Epot_Coulomb:.3f} eV")
        print("#" * 80)

        # calculating fragment mass segments
        low_mass_frag = math.ceil((LMCO + 0.1) / 100 * mass_prec_u / charge_prec_z)
        high_mass_frag = math.floor(mass_prec_u / charge_prec_z * 2)
        masslist_frag = np.linspace(low_mass_frag, high_mass_frag, number_mass_segments + 1)
        factor_list = np.linspace(0, 1, coulomb_factor_segments + 1)
        for coulomb_factor in factor_list:
            # creating result dictionary structure
            summed_Coulomb_factors[cutoff][precursormass][coulomb_factor] = {}
            summed_Coulomb_factors[cutoff][precursormass][coulomb_factor]["maxT"] = [[], [], []]
            summed_Coulomb_factors[cutoff][precursormass][coulomb_factor]["avgT"] = []
            summed_Coulomb_factors[cutoff][precursormass][coulomb_factor]["minT"] = [[], [], []]
            summed_Coulomb_factors[cutoff][precursormass][coulomb_factor]["lost%"] = []
            summed_Coulomb_factors[cutoff][precursormass][coulomb_factor]["cold%"] = []
            summed_Coulomb_factors[cutoff][precursormass][coulomb_factor]["lostavgT"] = []
            summed_Coulomb_factors[cutoff][precursormass][coulomb_factor]["trappedavgT"] = []
            summed_Coulomb_factors[cutoff][precursormass][coulomb_factor]["coldavgT"] = []
        summed_Coulomb_factors[cutoff][precursormass]["masslist"] = masslist_frag.copy()

        for index_fragmass, fragmentmass in enumerate(masslist_frag):

            # creating result dictionary structure
            fragmentation_factors[cutoff][precursormass][fragmentmass] = {}

            print("Finished: {:4.1f}%: low-mass cutoff: {} / {}, precursor mass: {} / {} "
                  "fragment {:3} / {:3} and from {} Coulomb factors finished: "
                  .format((index_cutoff * len(precursormass_to_calculate) * len(masslist_frag) + index_precmass
                           * len(masslist_frag) + index_fragmass)
                          / (len(LMCO_to_calculate)* len(precursormass_to_calculate) * len(masslist_frag)) * 100,
                          index_cutoff,
                          len(LMCO_to_calculate),
                          index_precmass,
                          len(precursormass_to_calculate),
                          index_fragmass, len(masslist_frag),
                          coulomb_factor_segments + 1), end="")


            mass_frag_u = fragmentmass  # in u
            mass_frag = mass_frag_u / AVOGADRO  # in kg
            charge_frag_z = 1  # in e
            charge_frag = charge_frag_z * ELECTRON  # in As
            zp_Coulomb_light = (2 * ELECTRON * Epot_Coulomb / (mass_prec + mass_frag) * mass_prec / mass_frag)**0.5
            zp_Coulomb_heavy = zp_Coulomb_light * mass_frag / mass_prec
            up_Coulomb_light = zp_Coulomb_light / (zmax_prec_fragmentation / 1000) * 2 / (2*PI * fRF * 1000)
            up_Coulomb_heavy = zp_Coulomb_heavy / (zmax_prec_fragmentation / 1000) * 2 / (2*PI * fRF * 1000)

            # determining fragment variables
            q_frag = q_prec * mass_prec_u / charge_prec_z / mass_frag_u * charge_frag_z
            a_frag = a_prec * mass_prec_u / charge_prec_z / mass_frag_u * charge_frag_z

            # calculating secular frequency of fragment beta_frag
            beta_frag = frequency(q=q_frag, a=a_frag)

            exp_coeff[FRAG][0] = (1.0, 1 * (beta_frag + 2 * 0), 1 * (beta_frag + 2 * 0)**2)
            sum_exp_coeff[FRAG] = [1.0, beta_frag, beta_frag**2]
            for n in range(harmonic_limit + 1):
                C = exp_coeff[FRAG][n][0] * (- q_frag) / ((beta_frag + 2 * (n + 1))**2 - a_frag - (q_frag**2
                                                        / ((beta_frag + 2 * (n + 2))**2 - a_frag - (q_frag**2
                                                         / ((beta_frag + 2 * (n + 3))**2 - a_frag - (q_frag**2
                                                          / ((beta_frag + 2 * (n + 4))**2 - a_frag - (q_frag**2
                                                           / ((beta_frag + 2 * (n + 5))**2 - a_frag)))))))))
                exp_coeff[FRAG][n + 1] = (C, C * (beta_frag + 2 * (n + 1)), C * (beta_frag + 2 * (n + 1))**2)
                sum_exp_coeff[FRAG][C2N] += exp_coeff[FRAG][n + 1][C2N]
                sum_exp_coeff[FRAG][C2N_TIMES_FREQ] += exp_coeff[FRAG][n + 1][C2N_TIMES_FREQ]
                sum_exp_coeff[FRAG][C2N_TIMES_FREQSQUARED] += exp_coeff[FRAG][n + 1][C2N_TIMES_FREQSQUARED]
            for n in range(0, -(harmonic_limit + 1), -1):
                C = exp_coeff[FRAG][n][0] * (- q_frag) / ((beta_frag + 2 * (n - 1))**2 - a_frag - (q_frag**2
                                                        / ((beta_frag + 2 * (n - 2))**2 - a_frag - (q_frag**2
                                                         / ((beta_frag + 2 * (n - 3))**2 - a_frag - (q_frag**2
                                                          / ((beta_frag + 2 * (n - 4))**2 - a_frag - (q_frag**2
                                                           / ((beta_frag + 2 * (n - 5))**2 - a_frag)))))))))
                exp_coeff[FRAG][n - 1] = (C, C * (beta_frag + 2 * (n - 1)), C * (beta_frag + 2 * (n - 1))**2)
                sum_exp_coeff[FRAG][C2N] += exp_coeff[FRAG][n - 1][C2N]
                sum_exp_coeff[FRAG][C2N_TIMES_FREQ] += exp_coeff[FRAG][n - 1][C2N_TIMES_FREQ]
                sum_exp_coeff[FRAG][C2N_TIMES_FREQSQUARED] += exp_coeff[FRAG][n - 1][C2N_TIMES_FREQSQUARED]
            exp_coeff_frag_sorted = sorted(list(exp_coeff[FRAG].items()))

            # calculating fragment maximum displacement umax, and average position and velocities
            aPosition_avg_frag = 0
            aPosition_squared_avg_frag = 0
            aVelocity_avg_frag = 0
            aVelocity_squared_avg_frag = 0
            aPosition_max_frag = 0
            aVelocity_max_frag = 0
            phi_segments = 0
            for phi in np.linspace(0, 90, phi_steps + 1):

                # using my approach for calculation: https://doi.org/10.1016/j.ijms.2021.116641
                aPosition_frag = 0.0
                aPosition_squared_frag = 0.0
                aVelocity_frag = 0.0
                aVelocity_squared_frag = 0.0
                for m in range(-harmonic_limit, harmonic_limit + 1, 1):
                    for o in range(-harmonic_limit, harmonic_limit + 1, 1):
                        aPosition_squared_frag += exp_coeff[FRAG][m][C2N] * exp_coeff[FRAG][o][C2N]\
                                                  * np.cos((m - o) * 2*phi / 360 * 2*PI)
                        aVelocity_squared_frag += exp_coeff[FRAG][m][C2N_TIMES_FREQ] * exp_coeff[FRAG][o][C2N_TIMES_FREQ]\
                                                  * np.cos((m - o) * 2*phi / 360 * 2*PI)
                aPosition_frag = aPosition_squared_frag**0.5
                aVelocity_frag = aVelocity_squared_frag**0.5
                if aPosition_frag > aPosition_max_frag:
                    aPosition_max_frag = aPosition_frag
                    phi_aPosition_max_frag = phi
                if aVelocity_frag > aVelocity_max_frag:
                    aVelocity_max_frag = aVelocity_frag
                    phi_aVelocity_max_frag = phi
                if phi == 0 or phi == 90:
                    aPosition_avg_frag += aPosition_frag
                    aPosition_squared_avg_frag += aPosition_squared_frag
                    aVelocity_avg_frag += aVelocity_frag
                    aVelocity_squared_avg_frag += aVelocity_squared_frag
                    phi_segments += 1
                else:
                    aPosition_avg_frag += 2 * aPosition_frag
                    aPosition_squared_avg_frag += 2 * aPosition_squared_frag
                    aVelocity_avg_frag += 2 * aVelocity_frag
                    aVelocity_squared_avg_frag += 2 * aVelocity_squared_frag
                    phi_segments += 2
            umax_frag = aPosition_max_frag
            upmax_frag = aVelocity_max_frag
            zpmax_frag = upmax_frag / umax_frag * z0 * 2*PI * fRF / 2
            uavg_frag = aPosition_avg_frag / phi_segments * 2 / PI
            upavg_frag = aVelocity_avg_frag / phi_segments * 2 / PI
            zpavg_frag = upavg_frag / umax_frag * z0 * 2*PI * fRF / 2
            uRMS_frag = (aPosition_squared_avg_frag / phi_segments)**0.5 / 2**0.5
            upRMS_frag = (aVelocity_squared_avg_frag / phi_segments)**0.5 / 2**0.5
            zpRMS_frag = upRMS_frag / umax_frag * z0 * 2*PI * fRF / 2

            T_CS_cold = zpRMS_prec / zpRMS_frag
            T_CS_loss = z0 / zmax_prec_fragmentation
            for index_coulomb_factor, coulomb_factor in enumerate(factor_list):

                # creating result dictionary structure
                fragmentation_factors[cutoff][precursormass][fragmentmass][coulomb_factor]\
                    = [[0, 0], [0, 0], [0, 0], [0, 0, 0], [1000000, 0, 0]]

                print(f" {index_coulomb_factor}", end="")

                for phi in np.linspace(0, 180, phi_steps):
                    u1_frag = 0
                    u2_frag = 0
                    up1_frag = 0
                    up2_frag = 0
                    for delta_prec in np.linspace(0, 360, delta_prec_steps):
                        uprec = 0
                        upprec = 0
                        for n in range(-harmonic_limit, harmonic_limit + 1, 1):
                            uprec += exp_coeff[PREC][n][C2N] * np.cos((delta_prec + 2*n*phi) / 360 * 2*PI)
                            upprec += -exp_coeff[PREC][n][C2N_TIMES_FREQ] * np.sin((delta_prec + 2*n*phi) / 360 * 2*PI)
                            if delta_prec == 0:
                                u1_frag += exp_coeff[FRAG][n][C2N]\
                                           * np.cos((beta_frag + 2 * n) * phi / 360 * 2*PI)
                                u2_frag += exp_coeff[FRAG][n][C2N]\
                                           * np.sin((beta_frag + 2 * n) * phi / 360 * 2*PI)
                                up1_frag += -exp_coeff[FRAG][n][C2N_TIMES_FREQ]\
                                            * np.sin((beta_frag + 2 * n) * phi / 360 * 2*PI)
                                up2_frag += exp_coeff[FRAG][n][C2N_TIMES_FREQ]\
                                            * np.cos((beta_frag + 2 * n) * phi / 360 * 2*PI)
                        u_frag = u1_frag + u2_frag
                        up_frag = up1_frag + up2_frag
                        if mass_frag <= mass_prec:
                            alpha1_frag = (uprec * up2_frag - (upprec + coulomb_factor * up_Coulomb_light) * u2_frag)\
                                          / (sum_exp_coeff[FRAG][C2N] * sum_exp_coeff[FRAG][C2N_TIMES_FREQ])
                            alpha2_frag = (u1_frag * (upprec + coulomb_factor * up_Coulomb_light) - up1_frag * uprec)\
                                          / (sum_exp_coeff[FRAG][C2N] * sum_exp_coeff[FRAG][C2N_TIMES_FREQ])
                            T_CS = (alpha1_frag**2 + alpha2_frag**2)**0.5 * umax_frag / umax_prec
                        else:
                            alpha1_frag = ((uprec * up2_frag - (upprec + coulomb_factor * up_Coulomb_heavy) * u2_frag)
                                           / (sum_exp_coeff[FRAG][C2N] * sum_exp_coeff[FRAG][C2N_TIMES_FREQ]))
                            alpha2_frag = ((u1_frag * (upprec + coulomb_factor * up_Coulomb_heavy) - up1_frag * uprec)
                                           / (sum_exp_coeff[FRAG][C2N] * sum_exp_coeff[FRAG][C2N_TIMES_FREQ]))
                            T_CS = (alpha1_frag**2 + alpha2_frag**2)**0.5 * umax_frag / umax_prec

                        # writing the results in the corresponding result list
                        if T_CS > fragmentation_factors[cutoff][precursormass][fragmentmass][coulomb_factor][MAX][VALUE]:
                            fragmentation_factors[cutoff][precursormass][fragmentmass][coulomb_factor][MAX][VALUE] = T_CS
                            fragmentation_factors[cutoff][precursormass][fragmentmass][coulomb_factor][MAX][PHI] = phi
                            fragmentation_factors[cutoff][precursormass][fragmentmass][coulomb_factor][MAX][DELTA] = delta_prec
                        if T_CS < fragmentation_factors[cutoff][precursormass][fragmentmass][coulomb_factor][MIN][VALUE]:
                            fragmentation_factors[cutoff][precursormass][fragmentmass][coulomb_factor][MIN][VALUE] = T_CS
                            fragmentation_factors[cutoff][precursormass][fragmentmass][coulomb_factor][MIN][PHI] = phi
                            fragmentation_factors[cutoff][precursormass][fragmentmass][coulomb_factor][MIN][DELTA] = delta_prec
                        if T_CS >= T_CS_loss:
                            fragmentation_factors[cutoff][precursormass][fragmentmass][coulomb_factor][LOST][SUM] += T_CS
                            fragmentation_factors[cutoff][precursormass][fragmentmass][coulomb_factor][LOST][COUNT] += 1
                        if T_CS < T_CS_loss:
                            fragmentation_factors[cutoff][precursormass][fragmentmass][coulomb_factor][TRAPPED][SUM] += T_CS
                            fragmentation_factors[cutoff][precursormass][fragmentmass][coulomb_factor][TRAPPED][COUNT] += 1
                        if T_CS <= T_CS_cold:
                            fragmentation_factors[cutoff][precursormass][fragmentmass][coulomb_factor][COLD][SUM] += T_CS
                            fragmentation_factors[cutoff][precursormass][fragmentmass][coulomb_factor][COLD][COUNT] += 1
                count_tot = phi_steps * delta_prec_steps
                if count_tot != fragmentation_factors[cutoff][precursormass][fragmentmass][coulomb_factor][LOST][COUNT]\
                        + fragmentation_factors[cutoff][precursormass][fragmentmass][coulomb_factor][TRAPPED][COUNT]:
                    print("There has to be an error somewhere in the count of Tfactors!")
                summed_Coulomb_factors[cutoff][precursormass][coulomb_factor]["maxT"][0].append(
                    fragmentation_factors[cutoff][precursormass][fragmentmass][coulomb_factor][MAX][VALUE])
                summed_Coulomb_factors[cutoff][precursormass][coulomb_factor]["maxT"][1].append(
                    fragmentation_factors[cutoff][precursormass][fragmentmass][coulomb_factor][MAX][PHI])
                summed_Coulomb_factors[cutoff][precursormass][coulomb_factor]["maxT"][2].append(
                    fragmentation_factors[cutoff][precursormass][fragmentmass][coulomb_factor][MAX][DELTA])
                summed_Coulomb_factors[cutoff][precursormass][coulomb_factor]["avgT"].append(
                    (fragmentation_factors[cutoff][precursormass][fragmentmass][coulomb_factor][LOST][SUM]
                     + fragmentation_factors[cutoff][precursormass][fragmentmass][coulomb_factor][TRAPPED][SUM]) / count_tot)
                summed_Coulomb_factors[cutoff][precursormass][coulomb_factor]["minT"][0].append(
                    fragmentation_factors[cutoff][precursormass][fragmentmass][coulomb_factor][MIN][VALUE])
                summed_Coulomb_factors[cutoff][precursormass][coulomb_factor]["minT"][1].append(
                    fragmentation_factors[cutoff][precursormass][fragmentmass][coulomb_factor][MIN][PHI])
                summed_Coulomb_factors[cutoff][precursormass][coulomb_factor]["minT"][2].append(
                    fragmentation_factors[cutoff][precursormass][fragmentmass][coulomb_factor][MIN][DELTA])
                summed_Coulomb_factors[cutoff][precursormass][coulomb_factor]["lost%"].append(
                    fragmentation_factors[cutoff][precursormass][fragmentmass][coulomb_factor][LOST][COUNT] / count_tot * 100)
                summed_Coulomb_factors[cutoff][precursormass][coulomb_factor]["cold%"].append(
                    fragmentation_factors[cutoff][precursormass][fragmentmass][coulomb_factor][COLD][COUNT] / count_tot * 100)
                if fragmentation_factors[cutoff][precursormass][fragmentmass][coulomb_factor][LOST][COUNT] == 0:
                    summed_Coulomb_factors[cutoff][precursormass][coulomb_factor]["lostavgT"].append(0)
                else:
                    summed_Coulomb_factors[cutoff][precursormass][coulomb_factor]["lostavgT"].append(
                        fragmentation_factors[cutoff][precursormass][fragmentmass][coulomb_factor][LOST][SUM]
                        / fragmentation_factors[cutoff][precursormass][fragmentmass][coulomb_factor][LOST][COUNT])
                if fragmentation_factors[cutoff][precursormass][fragmentmass][coulomb_factor][TRAPPED][COUNT] == 0:
                    summed_Coulomb_factors[cutoff][precursormass][coulomb_factor]["trappedavgT"].append(0)
                else:
                    summed_Coulomb_factors[cutoff][precursormass][coulomb_factor]["trappedavgT"].append(
                        fragmentation_factors[cutoff][precursormass][fragmentmass][coulomb_factor][TRAPPED][SUM]
                        / fragmentation_factors[cutoff][precursormass][fragmentmass][coulomb_factor][TRAPPED][COUNT])
                if fragmentation_factors[cutoff][precursormass][fragmentmass][coulomb_factor][COLD][COUNT] == 0:
                    summed_Coulomb_factors[cutoff][precursormass][coulomb_factor]["coldavgT"].append(0)
                else:
                    summed_Coulomb_factors[cutoff][precursormass][coulomb_factor]["coldavgT"].append(
                        fragmentation_factors[cutoff][precursormass][fragmentmass][coulomb_factor][COLD][SUM]
                        / fragmentation_factors[cutoff][precursormass][fragmentmass][coulomb_factor][COLD][COUNT])

            if index_coulomb_factor == coulomb_factor_segments:
                print(f" {coulomb_factor_segments + 1}")

print("#" * 80)
print("#" * 80)
print("#" * 80)
print("Finished calculation of fragmentation factors.")
print("Saving plots now.")
print("#" * 80)
print("#" * 80)
print("#" * 80)

########################################################################################################################
# saving calculations to file
directory = os.getcwd()
savefile_folder = directory + "Coulomb_calculation_results/"
file_extensions = [".txt", ".py"]
while True:
    desired_save = input('Please choose the filename to where the results will be written: [type "QUIT" to quit] ')
    if desired_save.upper() == "QUIT":
        sys.exit("User wanted to quit.")
    desired_save = "".join(x for x in desired_save if (x.isalnum() or x in "._-"))
    if desired_save == "" or desired_save[0].isalnum() == False or desired_save[-1].isalnum() == False:
        print("*" * 80)
        print("Filename cannot be empty and first and last position must be letter or digit.")
        print("*" * 80)
        continue
    else:
        exist = False
        savefiles = []
        date = datetime.now().strftime("%Y%m%d_%H%M%S")
        for extension in file_extensions:
            filename_save = savefile_folder + date + "_" + desired_save.lower() + extension
            savefiles.append(filename_save)
        for file in savefiles:
            if os.path.exists(file):
                exist = True
            if not os.path.exists(os.path.dirname(file)):
                try:
                    os.makedirs(os.path.dirname(file))
                except:
                    print("Could not create the savefile folder!")
                    continue
        if exist == True:
            print("*" * 80)
            print("*" * 10 + "One of the files exists already!")
            print("*" * 80)
            overwrite = input("Do you want to overwrite files? [type 'YES' to overwrite or any key to choose a new name] ").upper()
            if overwrite.upper() != "YES":
                continue
        for file in savefiles:
            with open(file, "w") as savefile:
                print(f"Ion Trap prameters:, r0 = {r0} mm, z0 = {z0} mm, fRF = {fRF} kHz", file=savefile)
                print("Calculational parameters:, harmonic_limit = {}, number_mass_segments = {}, phi_steps_phasespace = {}, "
                      "coulomb_factor_segments = {}, phi_steps= {}, delta_prec_steps= {}, LMCO_to_calculate = {}, "
                      "precursormass_to_calculate = {}"
                      .format(harmonic_limit, number_mass_segments, phi_steps_phasespace, coulomb_factor_segments, phi_steps,
                              delta_prec_steps, LMCO_to_calculate, precursormass_to_calculate), file=savefile)
                print("\n\n", file=savefile)

                for cutoff in summed_Coulomb_factors.keys():
                    for precmass in summed_Coulomb_factors[cutoff].keys():
                        print(f"Calculation values:, LMCO: {cutoff}, Precursor mass: {precmass}u", file=savefile)
                        print("X-axis: Fragment m/z, ", file=savefile, end="")
                        for fragmass in summed_Coulomb_factors[cutoff][precmass]["masslist"]:
                            print(f"{fragmass}, ", file=savefile, end="")
                        print("\n", file=savefile)
                        for cfactor in summed_Coulomb_factors[cutoff][precmass].keys():
                            if cfactor == "masslist":
                                continue
                            print(f"Coulomb repulsion: {cfactor}%, ", file=savefile)
                            for plottype in summed_Coulomb_factors[cutoff][precmass][cfactor].keys():
                                print(f"Y-axis: {plottype}, ", file=savefile, end="")
                                if plottype == "maxT":
                                    for data in summed_Coulomb_factors[cutoff][precmass][cfactor]["maxT"][0]:
                                        print(f"{data}, ", file=savefile, end="")
                                    print("", file=savefile)
                                elif plottype == "minT":
                                    for data in summed_Coulomb_factors[cutoff][precmass][cfactor]["minT"][0]:
                                        print(f"{data}, ", file=savefile, end="")
                                    print("", file=savefile)
                                else:
                                    for datapoint in summed_Coulomb_factors[cutoff][precmass][cfactor][plottype]:
                                        print(f"{datapoint}, ", file=savefile, end="")
                                    print("", file=savefile)
                        print("\n\n\n", file=savefile)
            print("#" * 80)
            print(f"Successfully saved to {file}")
        break

########################################################################################################################
# plotting the graphs
for cutoff in summed_Coulomb_factors.keys():
    for precursormass in summed_Coulomb_factors[cutoff].keys():

        # x-axis
        fragmassplot = summed_Coulomb_factors[cutoff][precursormass]["masslist"]
        legend = []
        for coulomb_factor in summed_Coulomb_factors[cutoff][precursormass].keys():

            # different y-axes
            if coulomb_factor != "masslist":
                maxTplot = summed_Coulomb_factors[cutoff][precursormass][coulomb_factor]["maxT"]
                avgTplot = summed_Coulomb_factors[cutoff][precursormass][coulomb_factor]["avgT"]
                minTplot = summed_Coulomb_factors[cutoff][precursormass][coulomb_factor]["minT"]
                lostplot = summed_Coulomb_factors[cutoff][precursormass][coulomb_factor]["lost%"]
                coldplot = summed_Coulomb_factors[cutoff][precursormass][coulomb_factor]["cold%"]
                lostavgTplot = summed_Coulomb_factors[cutoff][precursormass][coulomb_factor]["lostavgT"]
                trappedavgTplot = summed_Coulomb_factors[cutoff][precursormass][coulomb_factor]["trappedavgT"]
                coldavgTplot = summed_Coulomb_factors[cutoff][precursormass][coulomb_factor]["coldavgT"]

                # which graphs to plot
                plt.plot(fragmassplot, lostplot, marker="o", markersize=4)
                legend.append(f"{coulomb_factor*100:.1f}")
                plt.plot(fragmassplot, coldplot, marker="o", markersize=4)
                legend.append(f"{coulomb_factor*100:.1f}")
        plt.title(f"LMCO = {cutoff}%, Precursor m = {precursormass}u, lost% and cold%")
        plt.xlabel("Fragment mass (m/z)")
        plt.ylabel("Lost or Cold ions in %")
        plt.legend(legend, title="Coulomb repulsion in %")
        plt.ylim(-5, 105)
        plt.axvline(x=precursormass/charge_prec_z, color='gray', linestyle='dashed', linewidth=1)
        plt.show()

for cutoff in summed_Coulomb_factors.keys():
    for precursormass in summed_Coulomb_factors[cutoff].keys():

        # x-axis
        fragmassplot = summed_Coulomb_factors[cutoff][precursormass]["masslist"]
        legend = []
        for coulomb_factor in summed_Coulomb_factors[cutoff][precursormass].keys():

            # different y-axes
            if coulomb_factor != "masslist":
                maxTplot = summed_Coulomb_factors[cutoff][precursormass][coulomb_factor]["maxT"]
                avgTplot = summed_Coulomb_factors[cutoff][precursormass][coulomb_factor]["avgT"]
                minTplot = summed_Coulomb_factors[cutoff][precursormass][coulomb_factor]["minT"]
                lostplot = summed_Coulomb_factors[cutoff][precursormass][coulomb_factor]["lost%"]
                coldplot = summed_Coulomb_factors[cutoff][precursormass][coulomb_factor]["cold%"]
                lostavgTplot = summed_Coulomb_factors[cutoff][precursormass][coulomb_factor]["lostavgT"]
                trappedavgTplot = summed_Coulomb_factors[cutoff][precursormass][coulomb_factor]["trappedavgT"]
                coldavgTplot = summed_Coulomb_factors[cutoff][precursormass][coulomb_factor]["coldavgT"]

                # which graphs to plot
                plt.plot(fragmassplot, avgTplot, marker="o", markersize=4)
                legend.append(f"{coulomb_factor*100:.1f}")
        plt.title(f"LMCO = {cutoff}%, Precursor m = {precursormass}u, avgT")
        plt.xlabel("Fragment mass (m/z)")
        plt.ylabel("average Fragmentation Factor T")
        plt.legend(legend, title="Coulomb repulsion in %")
        plt.ylim(0, 2.5)
        plt.axvline(x=precursormass/charge_prec_z, color='gray', linestyle='dashed', linewidth=1)
        plt.show()

for cutoff in summed_Coulomb_factors.keys():
    for precursormass in summed_Coulomb_factors[cutoff].keys():

        # x-axis
        fragmassplot = summed_Coulomb_factors[cutoff][precursormass]["masslist"]
        legend = []
        for coulomb_factor in summed_Coulomb_factors[cutoff][precursormass].keys():

            # different y-axes
            if coulomb_factor != "masslist":
                maxTplot = summed_Coulomb_factors[cutoff][precursormass][coulomb_factor]["maxT"]
                avgTplot = summed_Coulomb_factors[cutoff][precursormass][coulomb_factor]["avgT"]
                minTplot = summed_Coulomb_factors[cutoff][precursormass][coulomb_factor]["minT"]
                lostplot = summed_Coulomb_factors[cutoff][precursormass][coulomb_factor]["lost%"]
                coldplot = summed_Coulomb_factors[cutoff][precursormass][coulomb_factor]["cold%"]
                lostavgTplot = summed_Coulomb_factors[cutoff][precursormass][coulomb_factor]["lostavgT"]
                trappedavgTplot = summed_Coulomb_factors[cutoff][precursormass][coulomb_factor]["trappedavgT"]
                coldavgTplot = summed_Coulomb_factors[cutoff][precursormass][coulomb_factor]["coldavgT"]

                # which graphs to plot
                plt.plot(fragmassplot, trappedavgTplot, marker="o", markersize=4)
                legend.append(f"{coulomb_factor*100:.1f}")
        plt.title(f"LMCO = {cutoff}%, Precursor m = {precursormass}u, trappedavgTplot")
        plt.xlabel("Fragment mass (m/z)")
        plt.ylabel("average Fragmentation Factor T (lost ions)")
        plt.legend(legend, title="Coulomb repulsion in %")
        plt.axvline(x=precursormass/charge_prec_z, color='gray', linestyle='dashed', linewidth=1)
        plt.show()
