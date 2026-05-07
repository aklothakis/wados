"""Paper parameters from Liu et al. 2019 Tables 1, 2 and 4."""

# Paper Table 1 — design parameters (SI: metres, degrees, radians)
PAPER_PARAMS = {
    "beta_deg":  13.0,
    "L_w":       6.000,     # m
    "W":         3.000,     # m
    "L_s":       0.300,     # m
    "y5":        0.1608,    # m  (tip upper-surface height)
    "z5":        1.500,     # m  (tip spanwise position = W/2)
    "y6":        1.608,     # m  (centreline upper-surface height)
    "z6":        0.0,       # m
    "delta5":    0.0,       # rad
    "delta6":    0.0,       # rad
    "Ma_center": 6.0,
    "Ma_tip":    13.0,
    "gamma":     1.4,
}

# Paper Table 2 — constant-q trajectory (q ~ 64.2 kPa)
PAPER_TRAJECTORY = [
    {"Ma":  6, "alpha": 0.0, "H_km": 25.0, "T": 221.6, "P": 2549.0, "rho": 4.01e-2, "q": 64.3e3, "a": 298.40},
    {"Ma":  8, "alpha": 0.0, "H_km": 28.8, "T": 225.3, "P": 1443.0, "rho": 2.22e-2, "q": 64.2e3, "a": 300.00},
    {"Ma": 10, "alpha": 0.0, "H_km": 31.8, "T": 228.3, "P":  916.0, "rho": 1.40e-2, "q": 64.1e3, "a": 301.97},
    {"Ma": 13, "alpha": 0.0, "H_km": 35.4, "T": 237.6, "P":  543.0, "rho": 7.96e-3, "q": 64.2e3, "a": 309.02},
]

# Paper Table 4 — reference geometric metrics for waverider_M6-M13
PAPER_REFERENCE_GEOMETRY = {
    "Vol_m3":   3.02,
    "S_wet_m2": 26.23,
    "S_p_m2":    9.59,
    "S_b_m2":    1.42,
    "eta":       0.0797,
}

# Paper Fig. 12 — approximate aerodynamic reference values (graph reading, +/-5%)
PAPER_REFERENCE_AERO = {
    6:  {"CL": 0.175, "CD": 0.77, "L_D": 4.4, "Cmz": 0.52, "Xcp": 0.678},
    8:  {"CL": 0.160, "CD": 0.81, "L_D": 5.0, "Cmz": 0.55, "Xcp": 0.679},
    10: {"CL": 0.130, "CD": 0.72, "L_D": 5.4, "Cmz": 0.48, "Xcp": 0.678},
    13: {"CL": 0.090, "CD": 0.52, "L_D": 5.8, "Cmz": 0.36, "Xcp": 0.678},
}

# Aerodynamic reference dimensions (paper Section 4.2)
REF_LENGTH_M = 6.0
REF_AREA_M2  = 1.0
MOMENT_REF   = (0.0, 0.0, 0.0)   # pitching moment taken about the nose

# Acceptance tolerances for validation (fractional deviation)
TOLERANCES = {
    "Vol_m3":   0.03,
    "S_wet_m2": 0.02,
    "S_p_m2":   0.01,
    "S_b_m2":   0.03,
    "eta":      0.03,
    "CL":       0.10,
    "CD":       0.10,
    "L_D":      0.10,
    "Cmz":      0.15,
    "Xcp":      0.05,
}
