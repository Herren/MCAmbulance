import numpy as np

def calculate_mcInvM(E, px, py, pz, clip_negative=True):
    """
    Compute the invariant mass m from four-momentum components.

    This function assumes a high-energy physics (HEP) convention where the
    four-vector is (E, p_x, p_y, p_z) and the Minkowski metric is (+, -, -, -):

        m^2 = E^2 - p_x^2 - p_y^2 - p_z^2
        m   = sqrt(m^2)

    Parameters
    ----------
    E, px, py, pz : array_like
        Arrays (or scalars) of energy and momentum components. Inputs must be
        broadcastable to a common shape (NumPy broadcasting rules apply).
        Units must be consistent (commonly natural units with c=1).
    clip_negative : bool, default True
        If True, negative values of m^2 caused by floating-point rounding are
        clipped to 0 before taking the square root. This avoids NaNs for values
        that should be physically ~0 (e.g., massless particles).

    Returns
    -------
    numpy.ndarray
        Array of invariant mass values `m` with the broadcasted shape of the
        inputs.

    Notes
    -----
    - If `clip_negative=False`, negative m^2 values will produce `nan` from
      `np.sqrt`.
    - This function does not perform unit conversion.

    Examples
    --------
    >>> m = calculate_mcInvM(E, px, py, pz)
    >>> m = calculate_mcInvM(df["mu_E"], df["mu_px"], df["mu_py"], df["mu_pz"])
    """
    E = np.asarray(E, dtype=float)
    px = np.asarray(px, dtype=float)
    py = np.asarray(py, dtype=float)
    pz = np.asarray(pz, dtype=float)

    m2 = E*E - px*px - py*py - pz*pz
    if clip_negative:
        m2 = np.maximum(m2, 0.0)

    return np.sqrt(m2)

def get_decay_string(bmeson, channel, lepton):
    """
    Construct a LaTeX math-mode string describing a semileptonic B-meson decay.

    This function combines the formatted strings for the B meson, the hadronic
    channel, and the lepton-neutrino final state into a single expression
    suitable for Matplotlib rendering (wrapped in `$...$`).

    Parameters
    ----------
    bmeson : str
        B-meson identifier. Supported values:
        - "b0" : neutral B meson (B^0)
        - "bp" : charged B meson (B^+)
    channel : str
        Hadronic channel identifier. Must be one of the values supported by
        `get_channel_string`.
    lepton : str
        Lepton identifier. Supported values:
        - "mu"  : muon channel (μ ν_μ)
        - "e"   : electron channel (e ν_e)
        - "tau" : tau channel (τ ν_τ)

    Returns
    -------
    str
        A LaTeX string wrapped in `$...$` (math mode), e.g.
        `"$B^0\\to D_0^{*}(\\to D\\,\\pi)\\,\\mu\\,\\nu_\\mu$"`.

    Raises
    ------
    ValueError
        If `bmeson`, `channel`, or `lepton` is not recognized.
    """
    b_meson_string = get_bmeson_string(bmeson)
    channel_string = get_channel_string(channel)
    lepton_string = get_lepton_string(lepton)

    return rf"${b_meson_string}\to {channel_string}\,{lepton_string}$"


def get_bmeson_string(bmeson):
    """
    Map a short B-meson identifier to a LaTeX representation.

    Parameters
    ----------
    bmeson : str
        B-meson identifier. Supported values:
        - "b0" : returns "B^0"
        - "bp" : returns "B^+"

    Returns
    -------
    str
        LaTeX string representing the B meson (without `$...$` wrapping).

    Raises
    ------
    ValueError
        If `bmeson` is not one of the supported identifiers.
    """
    if bmeson == "b0":
        return r"B^0"
    elif bmeson == "bp":
        return r"B^+"
    raise ValueError(f"Unknown B meson: {bmeson}")


def get_channel_string(channel):
    """
    Map an internal hadronic-channel identifier to a LaTeX decay-chain string.

    The returned string describes an excited charm resonance and its decay mode,
    formatted for LaTeX/Matplotlib mathtext. The string is not wrapped in `$...$`.

    Parameters
    ----------
    channel : str
        Channel identifier. Supported values include:
        - "d0_d_pi"
        - "d0_d_eta"
        - "d0_d_pipi"
        - "d0_dstar_pipi"
        - "d0_d_pi_isgw2"
        - "d1p_dstar_pi"
        - "d1p_dstar_eta"
        - "d1p_d_pipi"
        - "d1p_dstar_pipi"
        - "d1p_dstar_pi_isgw2"

    Returns
    -------
    str
        LaTeX string describing the resonance and decay, e.g.
        `"D_0^{*}(\\to D\\,\\pi)"`.

    Raises
    ------
    ValueError
        If `channel` is not one of the supported identifiers.
    """
    if channel in ["d0_d_pi", "d0_d_pi_isgw2"]:
        return r"D_0^{*}(\to D\,\pi)"
    elif channel == "d0_d_eta":
        return r"D_0^{*}(\to D\,\eta)"
    elif channel == "d0_d_pipi":
        return r"D_0^{*}(\to D\,\pi\pi)"
    elif channel == "d0_dstar_pipi":
        return r"D_0^{*}(\to D^{*}\,\pi\pi)"
    elif channel in ["d1p_dstar_pi", "d1p_dstar_pi_isgw2"]:
        return r"D_1'(\to D^{*}\,\pi)"
    elif channel == "d1p_dstar_eta":
        return r"D_1'(\to D^{*}\,\eta)"
    elif channel == "d1p_d_pipi":
        return r"D_1'(\to D\,\pi\pi)"
    elif channel == "d1p_dstar_pipi":
        return r"D_1'(\to D^{*}\,\pi\pi)"
    raise ValueError(f"Unknown channel: {channel}")


def get_lepton_string(lepton):
    """
    Map a lepton identifier to a LaTeX lepton+neutrino final-state string.

    Parameters
    ----------
    lepton : str
        Lepton identifier. Supported values:
        - "mu"  : muon + muon neutrino
        - "e"   : electron + electron neutrino
        - "tau" : tau + tau neutrino

    Returns
    -------
    str
        LaTeX string for the lepton-neutrino pair (without `$...$` wrapping),
        e.g. `"\\mu\\,\\nu_\\mu"`.

    Raises
    ------
    ValueError
        If `lepton` is not one of the supported identifiers.
    """
    if lepton == "mu":
        return r"\mu\,\nu_\mu"
    elif lepton == "e":
        return r"e\,\nu_e"
    elif lepton == "tau":
        return r"\tau\,\nu_\tau"
    raise ValueError(f"Unknown lepton: {lepton}")
