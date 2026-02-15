
import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.transforms as transforms  # kept (even if unused)

# -----------------------------------------------------------------------------
# Force Matplotlib defaults (default font, default styling)
# -----------------------------------------------------------------------------
plt.rcdefaults()

# =============================================================================
# Physical Constants
# =============================================================================
R_d = 287.05      # Gas constant for dry air (J/kg/K)
R_v = 461.5       # Gas constant for water vapor (J/kg/K)
c_p = 1005.7      # Specific heat at constant pressure (J/kg/K)
c_pv = 1870.0     # Specific heat of water vapor (J/kg/K)
L_v = 2.501e6     # Latent heat of vaporization (J/kg)
g = 9.81          # Gravitational acceleration (m/s²)
epsilon = R_d / R_v  # Ratio of gas constants (~0.622)
P_0 = 1000.0      # Reference pressure (hPa)
T_0 = 273.15      # 0°C in Kelvin

# =============================================================================
# Thermodynamic Functions
# =============================================================================

def celsius_to_kelvin(T_c):
    return T_c + T_0

def kelvin_to_celsius(T_k):
    return T_k - T_0

def potential_temperature(T, P):
    return T * (P_0 / P) ** (R_d / c_p)

def temperature_from_theta(theta, P):
    return theta * (P / P_0) ** (R_d / c_p)

def saturation_vapor_pressure(T):
    # Bolton (1980); T in °C, output in hPa
    return 6.112 * np.exp(17.67 * T / (T + 243.5))

def mixing_ratio(e, P):
    # e, P in hPa; output g/kg
    return 1000.0 * epsilon * e / (P - e)

def saturation_mixing_ratio(T, P):
    e_s = saturation_vapor_pressure(T)
    return mixing_ratio(e_s, P)

def dewpoint_from_mixing_ratio(w, P):
    # w in g/kg, P in hPa -> dewpoint °C
    w_kg = w / 1000.0
    e = (w_kg * P) / (epsilon + w_kg)
    ln = np.log(e / 6.112)
    return 243.5 * ln / (17.67 - ln)

def wet_bulb_temperature(T, Td, P):
    # iterative psychrometric approximation
    Tw = 0.5 * (T + Td)
    for _ in range(50):
        e_sw = saturation_vapor_pressure(Tw)
        w_sw = mixing_ratio(e_sw, P)

        e_d = saturation_vapor_pressure(Td)
        w = mixing_ratio(e_d, P)

        Tw_new = T - (L_v / c_p) * (w_sw - w) / 1000.0
        if abs(Tw_new - Tw) < 0.001:
            break
        Tw = 0.5 * (Tw + Tw_new)
    return Tw

def lcl_temperature(T, Td):
    T_k = celsius_to_kelvin(T)
    Td_k = celsius_to_kelvin(Td)
    T_lcl_k = 1.0 / (1.0 / (Td_k - 56.0) + np.log(T_k / Td_k) / 800.0) + 56.0
    return kelvin_to_celsius(T_lcl_k)

def lcl_pressure(T, Td, P):
    T_lcl = lcl_temperature(T, Td)
    T_k = celsius_to_kelvin(T)
    T_lcl_k = celsius_to_kelvin(T_lcl)
    return P * (T_lcl_k / T_k) ** (c_p / R_d)

def moist_adiabat(T_start, P_levels):
    """
    Pseudoadiabatic-ish stepping (simple) from T_start at P_levels[0].
    T_start in °C; P_levels in hPa.
    """
    T = np.zeros_like(P_levels, dtype=float)
    T[0] = T_start

    for i in range(1, len(P_levels)):
        dP = P_levels[i] - P_levels[i - 1]  # hPa step
        T_k = celsius_to_kelvin(T[i - 1])
        P = P_levels[i - 1]

        w_s = saturation_mixing_ratio(T[i - 1], P) / 1000.0  # kg/kg

        numerator = 1.0 + (L_v * w_s) / (R_d * T_k)
        denominator = 1.0 + (L_v**2 * w_s * epsilon) / (c_p * R_d * T_k**2)
        gamma_m = (g / c_p) * numerator / denominator

        dT = gamma_m * (R_d * T_k / (g * P)) * dP
        T[i] = T[i - 1] + dT

        T[i] = max(T[i], -100.0)

    return T

# =============================================================================
# Tephigram Coordinate Transformation
# =============================================================================

class TephigramTransform:
    """
    Transform between (T, P) and tephigram (x, y) coordinates.

    y = -ln(P/P0)*scale
    x = T + y*tan(rotation)
    """

    def __init__(self, rotation_angle=45.0, y_scale=30.0):
        self.rotation = np.radians(rotation_angle)
        self.y_scale = float(y_scale)

    def to_tephigram(self, T, P):
        T = np.atleast_1d(np.asarray(T, dtype=float))
        P = np.atleast_1d(np.asarray(P, dtype=float))
        T, P = np.broadcast_arrays(T, P)

        y = -np.log(P / P_0) * self.y_scale
        x = T + y * np.tan(self.rotation)
        return x, y

    def from_tephigram(self, x, y):
        T = x - y * np.tan(self.rotation)
        P = P_0 * np.exp(-y / self.y_scale)
        return T, P

# =============================================================================
# Sounding Data Parser
# =============================================================================

def parse_sounding_file(filepath):
    """
    Parse a University of Wyoming-style sounding file.
    Returns arrays: pressure(hPa), temperature(°C), dewpoint(°C),
                    wind_dir(deg), wind_speed(knots)
    """

    # read in data skipping the first 4 rows as this contains the header information 
    data = np.loadtxt(filepath,skiprows=5)

    p = data[:,0]
    Ta = data[:,2]
    Td = data[:,3]
    Wd = data[:,6]
    Ws = data[:,7]

    bad = (p == -999)|(Ta == -999)|(Td == -999)|(Wd == -999)|(Ws == -999)

    with open(filepath, "r") as fobj:
        contents = fobj.readlines()
    launchtime = contents[0].strip().split('at')[-1]
    launchtime = datetime.datetime.strptime(launchtime," %HZ %d %b %Y")
    return (
        np.ma.masked_array(p, mask=bad).compressed(),
        np.ma.masked_array(Ta, mask=bad).compressed(),
        np.ma.masked_array(Td, mask=bad).compressed(),
        np.ma.masked_array(Wd, mask=bad).compressed(),
        np.ma.masked_array(Ws, mask=bad).compressed(),
        launchtime,
    )

# =============================================================================
# Wind Barb Drawing (BIGGER)
# =============================================================================

def draw_wind_barb(ax, x, y, direction, speed, size=1.8, linewidth=1.8):
    """
    Draw a larger wind barb at position (x, y).
    direction: meteorological degrees (wind FROM)
    speed: knots
    size: overall scaling
    """
    x = float(np.atleast_1d(x)[0])
    y = float(np.atleast_1d(y)[0])
    direction = float(direction)
    speed = float(speed)

    # Convert meteorological "from" to math angle
    angle = np.radians(270.0 - direction)
    dx = np.cos(angle)
    dy = np.sin(angle)

    # Main staff
    staff_len = size * 2.2
    ax.plot([x, x + staff_len * dx], [y, y + staff_len * dy],
            "k-", linewidth=linewidth, clip_on=True)

    remaining_speed = speed
    barb_pos = 0.92
    barb_spacing = 0.14

    # Pennants (50 kt)
    while remaining_speed >= 50.0:
        pos = barb_pos * staff_len
        x1 = x + pos * dx
        y1 = y + pos * dy

        perp_dx = -dy * size * 0.55
        perp_dy =  dx * size * 0.55

        x2 = x1 + perp_dx
        y2 = y1 + perp_dy

        x3 = x + (barb_pos - 0.16) * staff_len * dx
        y3 = y + (barb_pos - 0.16) * staff_len * dy

        tri = plt.Polygon([[x1, y1], [x2, y2], [x3, y3]],
                          facecolor="black", edgecolor="black")
        ax.add_patch(tri)

        remaining_speed -= 50.0
        barb_pos -= 0.16

    # Full barbs (10 kt)
    while remaining_speed >= 10.0:
        pos = barb_pos * staff_len
        x1 = x + pos * dx
        y1 = y + pos * dy

        perp_dx = -dy * size * 0.48
        perp_dy =  dx * size * 0.48

        ax.plot([x1, x1 + perp_dx], [y1, y1 + perp_dy],
                "k-", linewidth=linewidth, clip_on=True)

        remaining_speed -= 10.0
        barb_pos -= barb_spacing

    # Half barb (5 kt)
    if remaining_speed >= 5.0:
        pos = barb_pos * staff_len
        x1 = x + pos * dx
        y1 = y + pos * dy

        perp_dx = -dy * size * 0.28
        perp_dy =  dx * size * 0.28

        ax.plot([x1, x1 + perp_dx], [y1, y1 + perp_dy],
                "k-", linewidth=linewidth, clip_on=True)

# =============================================================================
# Main Tephigram Plotting Function
# =============================================================================

def plot_tephigram(
    pressure,
    temperature,
    dewpoint,
    wind_dir=None,
    wind_speed=None,
    location="Sample Location",
    date="28.01.2026.",
    time="12:00",
    save_path=None,
):
    # -----------------------------
    # Font sizes (bigger)
    # -----------------------------
    TICK_LABEL_FS = 14
    AXIS_LABEL_FS = 16
    TITLE_FS = 15
    LEGEND_FS = 10.5
    SMALL_ANNOT_FS = 10.5
    DATETIME_FS = 13

    transform = TephigramTransform(rotation_angle=45.0, y_scale=30.0)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    T_min, T_max = -70.0, 45.0
    P_min, P_max = 100.0, 1050.0

    P_levels = np.array([1000, 900, 800, 700, 600, 500, 400, 300, 250, 200, 150, 100], dtype=float)

    # =========================================================================
    # Background Grid
    # =========================================================================

    # Isobars
    for P in P_levels:
        T_range = np.linspace(T_min - 50, T_max + 50, 100)
        x_vals, y_vals = transform.to_tephigram(T_range, P)
        ax.plot(x_vals, y_vals, color="#505050", linewidth=0.5, alpha=0.6)

    # Isotherms
    isotherm_values = np.arange(-80, 50, 10)
    for T in isotherm_values:
        P_range = np.logspace(np.log10(P_min), np.log10(P_max), 100)
        x_vals, y_vals = transform.to_tephigram(T, P_range)
        if T == 0:
            ax.plot(x_vals, y_vals, color="#00DDDD", linewidth=1.8, linestyle="--", alpha=0.95, zorder=2)
        else:
            ax.plot(x_vals, y_vals, color="#707070", linewidth=0.4, alpha=0.5)

    # Dry adiabats (theta)
    theta_values = np.arange(-30, 180, 10)
    for theta in theta_values:
        P_range = np.logspace(np.log10(P_min), np.log10(P_max), 100)
        theta_K = celsius_to_kelvin(theta)
        T_vals = kelvin_to_celsius(temperature_from_theta(theta_K, P_range))
        x_vals, y_vals = transform.to_tephigram(T_vals, P_range)
        ax.plot(x_vals, y_vals, color="#808080", linewidth=0.5, alpha=0.55)

    # Moist adiabats
    moist_start_temps = np.arange(-40, 45, 5)
    for T_start in moist_start_temps:
        P_range = np.linspace(P_max, P_min, 200)
        T_moist = moist_adiabat(T_start, P_range)
        x_vals, y_vals = transform.to_tephigram(T_moist, P_range)
        ax.plot(x_vals, y_vals, color="#00CED1", linewidth=0.6, alpha=0.5)

    # Mixing ratio lines + labels
    mixing_ratios = [0.4, 1, 2, 3, 5, 8, 12, 16, 20, 24, 32, 40]
    for w in mixing_ratios:
        P_range = np.linspace(P_max, 400.0, 100)
        try:
            T_vals = np.array([dewpoint_from_mixing_ratio(w, P) for P in P_range], dtype=float)
            valid = (~np.isnan(T_vals)) & (T_vals > -80) & (T_vals < 60)
            if np.sum(valid) > 10:
                x_vals, y_vals = transform.to_tephigram(T_vals[valid], P_range[valid])
                ax.plot(x_vals, y_vals, color="#DB7093", linewidth=0.5, linestyle="--", alpha=0.6)

                # label at ~1000 hPa
                try:
                    T_at_1000 = dewpoint_from_mixing_ratio(w, 1000.0)
                    x_label, y_label = transform.to_tephigram(T_at_1000, 1000.0)
                    if -60 < float(x_label[0]) < 45:
                        ax.text(
                            float(x_label[0]),
                            float(y_label[0]) - 0.8,
                            f"{w}",
                            fontsize=SMALL_ANNOT_FS,
                            color="#DB7093",
                            ha="center",
                            va="top",
                            alpha=0.9,
                        )
                except Exception:
                    pass
        except Exception:
            pass

    # =========================================================================
    # Parcel / LCL / CAPE / CIN (geometry only)
    # =========================================================================

    T_sfc = float(temperature[0])
    Td_sfc = float(dewpoint[0])
    P_sfc = float(pressure[0])

    T_lcl = lcl_temperature(T_sfc, Td_sfc)
    P_lcl = lcl_pressure(T_sfc, Td_sfc, P_sfc)

    parcel_P = np.concatenate([np.linspace(P_sfc, P_lcl, 30), np.linspace(P_lcl, 100.0, 100)])
    parcel_T = np.zeros_like(parcel_P, dtype=float)

    # Dry up to LCL
    theta_sfc = potential_temperature(celsius_to_kelvin(T_sfc), P_sfc)
    below_lcl = parcel_P >= P_lcl
    parcel_T[below_lcl] = kelvin_to_celsius(temperature_from_theta(theta_sfc, parcel_P[below_lcl]))

    # Moist above LCL
    above_lcl = parcel_P < P_lcl
    P_above = parcel_P[above_lcl]
    T_moist = moist_adiabat(T_lcl, np.concatenate([[P_lcl], P_above]))
    parcel_T[above_lcl] = T_moist[1:]

    # Interpolate environment to parcel levels (ensure x increasing for interp)
    env_T_interp = np.interp(parcel_P, pressure[::-1], temperature[::-1])
    T_diff = parcel_T - env_T_interp

    # Find LFC and EL by sign changes above LCL
    lfc_idx = None
    for i in range(len(T_diff) - 1):
        if parcel_P[i] < P_lcl and T_diff[i] <= 0 and T_diff[i + 1] > 0:
            lfc_idx = i
            break

    el_idx = None
    if lfc_idx is not None:
        for i in range(lfc_idx, len(T_diff) - 1):
            if T_diff[i] > 0 and T_diff[i + 1] <= 0:
                el_idx = i
                break

    # CIN shading
    if lfc_idx is not None:
        cin_mask = (parcel_P >= parcel_P[lfc_idx]) & (parcel_P <= P_lcl) & (T_diff < 0)
        if np.any(cin_mask):
            idx = np.where(cin_mask)[0]
            if idx.size > 2:
                cin_P = parcel_P[idx]
                cin_parcel_T = parcel_T[idx]
                cin_env_T = env_T_interp[idx]

                x_parcel, y_parcel = transform.to_tephigram(cin_parcel_T, cin_P)
                x_env, y_env = transform.to_tephigram(cin_env_T, cin_P)

                poly_x = np.concatenate([x_parcel, x_env[::-1]])
                poly_y = np.concatenate([y_parcel, y_env[::-1]])

                ax.add_patch(
                    Polygon(list(zip(poly_x, poly_y)), facecolor="#87CEEB", alpha=0.5, edgecolor="none", zorder=3)
                )

    # CAPE shading
    if lfc_idx is not None and el_idx is not None:
        cape_mask = (parcel_P <= parcel_P[lfc_idx]) & (parcel_P >= parcel_P[el_idx]) & (T_diff > 0)
        if np.any(cape_mask):
            idx = np.where(cape_mask)[0]
            if idx.size > 2:
                cape_P = parcel_P[idx]
                cape_parcel_T = parcel_T[idx]
                cape_env_T = env_T_interp[idx]

                x_parcel, y_parcel = transform.to_tephigram(cape_parcel_T, cape_P)
                x_env, y_env = transform.to_tephigram(cape_env_T, cape_P)

                poly_x = np.concatenate([x_parcel, x_env[::-1]])
                poly_y = np.concatenate([y_parcel, y_env[::-1]])

                ax.add_patch(
                    Polygon(list(zip(poly_x, poly_y)), facecolor="#FFB0B0", alpha=0.55, edgecolor="none", zorder=3)
                )

    # =========================================================================
    # Profiles
    # =========================================================================

    x_T, y_T = transform.to_tephigram(temperature, pressure)
    ax.plot(x_T, y_T, color="#DD0000", linewidth=2.8, label="Air temperature", zorder=5)

    x_Td, y_Td = transform.to_tephigram(dewpoint, pressure)
    ax.plot(x_Td, y_Td, color="#0000CC", linewidth=2.8, label="Dewpoint temperature", zorder=5)

    x_parcel, y_parcel = transform.to_tephigram(parcel_T, parcel_P)
    ax.plot(x_parcel, y_parcel, color="black", linewidth=2.2, label="Air parcel profile", zorder=4)

    wet_bulb = np.array([wet_bulb_temperature(T, Td, P) for T, Td, P in zip(temperature, dewpoint, pressure)],
                        dtype=float)
    x_wb, y_wb = transform.to_tephigram(wet_bulb, pressure)
    ax.plot(x_wb, y_wb, color="#00CCCC", linewidth=1.8, label="Wetbulb temperature", zorder=4)

    # LCL marker
    x_lcl, y_lcl = transform.to_tephigram(T_lcl, P_lcl)
    ax.scatter([float(x_lcl[0])], [float(y_lcl[0])], color="#666666", s=90, marker="o",
               zorder=6, label="LCL", edgecolors="#333333", linewidths=1)

    # =========================================================================
    # Wind Barbs (EVERY 5th, BIGGER)
    # =========================================================================

    if wind_dir is not None and wind_speed is not None:
        x_barb_base = 44.0

        # Plot every 5th point (0,5,10,...)
        indices = np.arange(0, len(pressure), 5, dtype=int)

        for i in indices:
            P = float(pressure[i])
            if P_min <= P <= P_max:
                _, y_pos = transform.to_tephigram(0.0, P)
                if float(wind_speed[i]) > 0.0:
                    draw_wind_barb(
                        ax,
                        x_barb_base,
                        float(y_pos[0]),
                        float(wind_dir[i]),
                        float(wind_speed[i]),
                        size=1.8,
                        linewidth=1.8,
                    )

    # =========================================================================
    # Axis labels / ticks (REMOVE TICK MARKS)
    # =========================================================================

    _, y_bottom = transform.to_tephigram(0.0, P_max)
    _, y_top = transform.to_tephigram(0.0, P_min)

    ax.set_xlim(-65, 48)
    ax.set_ylim(float(y_bottom[0]) - 0.5, float(y_top[0]) + 0.5)

    y_ticks, y_labels = [], []
    for P in P_levels:
        _, y = transform.to_tephigram(0.0, P)
        y_ticks.append(float(y[0]))
        y_labels.append(f"{int(P)}")

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=TICK_LABEL_FS)
    ax.set_ylabel("Pressure [hPa]", fontsize=AXIS_LABEL_FS)

    ax.set_xlabel("Potential temperature [°C]", fontsize=AXIS_LABEL_FS)
    x_ticks = np.arange(-60, 50, 10)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{t}" for t in x_ticks], fontsize=TICK_LABEL_FS)

    # Remove tick marks (keep labels)
    ax.minorticks_off()
    ax.tick_params(axis="both", which="both", labelsize=TICK_LABEL_FS, length=0, width=0)

    for spine in ax.spines.values():
        spine.set_linewidth(1.6)
        spine.set_color("black")

    # =========================================================================
    # Legend
    # =========================================================================

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color="#DD0000", linewidth=2.5, label="Air Temperature"),
        Line2D([0], [0], color="#0000CC", linewidth=2.5, label="Dewpoint Temperature"),
        Line2D([0], [0], color="#00DDDD", linestyle="--", linewidth=1.5, label="0°C Isotherm"),
        Line2D([0], [0], color="#808080", linewidth=0.8, label="Dry Adiabats"),
        Line2D([0], [0], color="#00CED1", linewidth=0.8, label="Moist Adiabats"),
        Line2D([0], [0], color="#DB7093", linestyle="--", linewidth=0.8, label="Mixing Ratio (g kg$^{-1}$)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#666666",
               markersize=8, markeredgecolor="#333333", label="LCL Temperature"),
        Line2D([0], [0], color="black", linewidth=2, label="Air parcel profile"),
        Line2D([0], [0], color="#00CCCC", linewidth=1.5, label="Wetbulb Temperature"),
        Patch(facecolor="#87CEEB", alpha=0.5, label="CIN"),
        Patch(facecolor="#FFB0B0", alpha=0.55, label="CAPE"),
    ]

    legend = ax.legend(
        handles=legend_elements,
        loc="lower left",
        fontsize=LEGEND_FS,
        framealpha=0.95,
        edgecolor="#888888",
        fancybox=False,
        borderpad=0.7,
        labelspacing=0.5,
    )
    legend.get_frame().set_linewidth(0.9)

    # =========================================================================
    # Title and Date/Time box
    # =========================================================================

    ax.set_title(
        f"Vertical Atmospheric Profile • {location}",#\n"f"Coordinates: {lat:.2f}°N, {lon:.2f}°E • Tephigram",
        fontsize=TITLE_FS,
        fontweight="bold",
        pad=16,
        loc="left",
    )

    bbox_props = dict(facecolor="black", alpha=0.9)
    ax.text(
        0.98,
        0.98,
        f"{date} | {time}",
        transform=ax.transAxes,
        fontsize=DATETIME_FS,
        va="top",
        ha="right",
        bbox=bbox_props,
        color="white",
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.09, right=0.95)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none")
        print(f"Tephigram saved to: {save_path}")
        plt.close()
    else:
        plt.show()

    return fig, ax

