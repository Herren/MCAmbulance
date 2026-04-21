#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2026 Florian Herren
# Copyright (c) 2026 Raynette van Tonder
#
# This file is part of MCAmbulance.
# 
# MCAmbulance is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3
# of the License, or (at your option) any later version.
#
# MCAmbulance is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with MCAmbulance.
# If not, see <https://www.gnu.org/licenses/>. 

from copy import deepcopy
from .conf import kin_conf, hqet_conf, isgw2_conf, bcl_conf
import numpy as np

broad_isgw2_conf = isgw2_conf()
broad_isgw2_conf.mbx = 2.4675


broad_llsw_conf = hqet_conf()
broad_llsw_conf.params = np.array([0.68, -0.2, 0.3])
broad_llsw_conf.m_b = 4.2
broad_llsw_conf.m_c = 1.4
broad_llsw_conf.L = 0.4
broad_llsw_conf.Lps = 0.76


ff_conf_dict = {}
ff_conf_dict[('d0')] = broad_llsw_conf
ff_conf_dict[('d1p')] = broad_llsw_conf
ff_conf_dict[('d0_isgw2')] = broad_isgw2_conf
ff_conf_dict[('d1p_isgw2')] = broad_isgw2_conf

m_e = 0.51099895e-3
m_mu = 105.65837e-3
m_tau = 1776.86e-3

m_pi_pm = 139.57039e-3
m_pi_0 = 134.9768e-3
m_eta = 547.857e-3 # EvtGen includes eta width, avoid nans by shifting value in the last digit

m_D_pm = 1869.65e-3
m_D_0 = 1864.83e-3
m_Dstar_pm = 2010.26e-3
m_Dstar_0 = 2006.85e-3

m_D0_pm = 2349.0e-3
gam_D0_pm = 221.0e-3
m_D0_0 = 2300.0e-3
gam_D0_0 = 274.0e-3

m_D1p_pm = 2427.0e-3
gam_D1p_pm = 314.0e-3
m_D1p_0 = 2427.0e-3
gam_D1p_0 = 314.0e-3

m_B_pm = 5279.34e-3
m_B_0 = 5279.65e-3


bp_d0_e_basf2_kin_conf = kin_conf()
bp_d0_e_basf2_kin_conf.m_1 = m_B_pm
bp_d0_e_basf2_kin_conf.m_2 = m_D_0
bp_d0_e_basf2_kin_conf.m_3 = m_pi_0
bp_d0_e_basf2_kin_conf.m_l = m_e
bp_d0_e_basf2_kin_conf.m_nom = m_D0_0
bp_d0_e_basf2_kin_conf.g_nom = gam_D0_0
bp_d0_e_basf2_kin_conf.l = 0

bp_d0_mu_basf2_kin_conf = deepcopy(bp_d0_e_basf2_kin_conf)
bp_d0_mu_basf2_kin_conf.m_l = m_mu
bp_d0_tau_basf2_kin_conf = deepcopy(bp_d0_e_basf2_kin_conf)
bp_d0_tau_basf2_kin_conf.m_l = m_tau


bp_d0_e_eta_basf2_kin_conf = kin_conf()
bp_d0_e_eta_basf2_kin_conf.m_1 = m_B_pm
bp_d0_e_eta_basf2_kin_conf.m_2 = m_D_0
bp_d0_e_eta_basf2_kin_conf.m_3 = m_eta
bp_d0_e_eta_basf2_kin_conf.m_l = m_e
bp_d0_e_eta_basf2_kin_conf.m_nom = m_D0_0
bp_d0_e_eta_basf2_kin_conf.g_nom = gam_D0_0
bp_d0_e_eta_basf2_kin_conf.l = 0
bp_d0_e_eta_basf2_kin_conf.subthreshold = True

bp_d0_mu_eta_basf2_kin_conf = deepcopy(bp_d0_e_eta_basf2_kin_conf)
bp_d0_mu_eta_basf2_kin_conf.m_l = m_mu
bp_d0_tau_eta_basf2_kin_conf = deepcopy(bp_d0_e_eta_basf2_kin_conf)
bp_d0_tau_eta_basf2_kin_conf.m_l = m_tau


b0_d0_e_basf2_kin_conf = kin_conf()
b0_d0_e_basf2_kin_conf.m_1 = m_B_0
b0_d0_e_basf2_kin_conf.m_2 = m_D_0
b0_d0_e_basf2_kin_conf.m_3 = m_pi_pm
b0_d0_e_basf2_kin_conf.m_l = m_e
b0_d0_e_basf2_kin_conf.m_nom = m_D0_pm
b0_d0_e_basf2_kin_conf.g_nom = gam_D0_pm
b0_d0_e_basf2_kin_conf.l = 0

b0_d0_mu_basf2_kin_conf = deepcopy(b0_d0_e_basf2_kin_conf)
b0_d0_mu_basf2_kin_conf.m_l = m_mu
b0_d0_tau_basf2_kin_conf = deepcopy(b0_d0_e_basf2_kin_conf)
b0_d0_tau_basf2_kin_conf.m_l = m_tau


b0_d0_e_eta_basf2_kin_conf = kin_conf()
b0_d0_e_eta_basf2_kin_conf.m_1 = m_B_0
b0_d0_e_eta_basf2_kin_conf.m_2 = m_D_pm
b0_d0_e_eta_basf2_kin_conf.m_3 = m_eta
b0_d0_e_eta_basf2_kin_conf.m_l = m_e
b0_d0_e_eta_basf2_kin_conf.m_nom = m_D0_pm
b0_d0_e_eta_basf2_kin_conf.g_nom = gam_D0_pm
b0_d0_e_eta_basf2_kin_conf.l = 0
b0_d0_e_eta_basf2_kin_conf.subthreshold = True

b0_d0_mu_eta_basf2_kin_conf = deepcopy(b0_d0_e_eta_basf2_kin_conf)
b0_d0_mu_eta_basf2_kin_conf.m_l = m_mu
b0_d0_tau_eta_basf2_kin_conf = deepcopy(b0_d0_e_eta_basf2_kin_conf)
b0_d0_tau_eta_basf2_kin_conf.m_l = m_tau


bp_d1p_e_basf2_kin_conf = kin_conf()
bp_d1p_e_basf2_kin_conf.m_1 = m_B_pm
bp_d1p_e_basf2_kin_conf.m_2 = m_Dstar_0
bp_d1p_e_basf2_kin_conf.m_3 = m_pi_0
bp_d1p_e_basf2_kin_conf.m_l = m_e
bp_d1p_e_basf2_kin_conf.m_nom = m_D1p_0
bp_d1p_e_basf2_kin_conf.g_nom = gam_D1p_0
bp_d1p_e_basf2_kin_conf.l = 0

bp_d1p_mu_basf2_kin_conf = deepcopy(bp_d1p_e_basf2_kin_conf)
bp_d1p_mu_basf2_kin_conf.m_l = m_mu
bp_d1p_tau_basf2_kin_conf = deepcopy(bp_d1p_e_basf2_kin_conf)
bp_d1p_tau_basf2_kin_conf.m_l = m_tau


b0_d1p_e_basf2_kin_conf = kin_conf()
b0_d1p_e_basf2_kin_conf.m_1 = m_B_0
b0_d1p_e_basf2_kin_conf.m_2 = m_Dstar_pm
b0_d1p_e_basf2_kin_conf.m_3 = m_pi_0
b0_d1p_e_basf2_kin_conf.m_l = m_e
b0_d1p_e_basf2_kin_conf.m_nom = m_D1p_pm
b0_d1p_e_basf2_kin_conf.g_nom = gam_D1p_pm
b0_d1p_e_basf2_kin_conf.l = 0

b0_d1p_mu_basf2_kin_conf = deepcopy(b0_d1p_e_basf2_kin_conf)
b0_d1p_mu_basf2_kin_conf.m_l = m_mu
b0_d1p_tau_basf2_kin_conf = deepcopy(b0_d1p_e_basf2_kin_conf)
b0_d1p_tau_basf2_kin_conf.m_l = m_tau


bp_d1p_e_eta_basf2_kin_conf = kin_conf()
bp_d1p_e_eta_basf2_kin_conf.m_1 = m_B_pm
bp_d1p_e_eta_basf2_kin_conf.m_2 = m_Dstar_0
bp_d1p_e_eta_basf2_kin_conf.m_3 = m_eta
bp_d1p_e_eta_basf2_kin_conf.m_l = m_e
bp_d1p_e_eta_basf2_kin_conf.m_nom = m_D1p_0
bp_d1p_e_eta_basf2_kin_conf.g_nom = gam_D1p_0
bp_d1p_e_eta_basf2_kin_conf.l = 0
bp_d1p_e_eta_basf2_kin_conf.subthreshold = True

bp_d1p_mu_eta_basf2_kin_conf = deepcopy(bp_d1p_e_eta_basf2_kin_conf)
bp_d1p_mu_eta_basf2_kin_conf.m_l = m_mu
bp_d1p_tau_eta_basf2_kin_conf = deepcopy(bp_d1p_e_eta_basf2_kin_conf)
bp_d1p_tau_eta_basf2_kin_conf.m_l = m_tau


b0_d1p_e_eta_basf2_kin_conf = kin_conf()
b0_d1p_e_eta_basf2_kin_conf.m_1 = m_B_0
b0_d1p_e_eta_basf2_kin_conf.m_2 = m_Dstar_pm
b0_d1p_e_eta_basf2_kin_conf.m_3 = m_eta
b0_d1p_e_eta_basf2_kin_conf.m_l = m_e
b0_d1p_e_eta_basf2_kin_conf.m_nom = m_D1p_pm
b0_d1p_e_eta_basf2_kin_conf.g_nom = gam_D1p_pm
b0_d1p_e_eta_basf2_kin_conf.l = 0
b0_d1p_e_eta_basf2_kin_conf.subthreshold = True

b0_d1p_mu_eta_basf2_kin_conf = deepcopy(b0_d1p_e_eta_basf2_kin_conf)
b0_d1p_mu_eta_basf2_kin_conf.m_l = m_mu
b0_d1p_tau_eta_basf2_kin_conf = deepcopy(b0_d1p_e_eta_basf2_kin_conf)
b0_d1p_tau_eta_basf2_kin_conf.m_l = m_tau


bp_d1p_e_pipi_basf2_kin_conf = kin_conf()
bp_d1p_e_pipi_basf2_kin_conf.m_1 = m_B_pm
bp_d1p_e_pipi_basf2_kin_conf.m_2 = m_D_0
bp_d1p_e_pipi_basf2_kin_conf.m_3 = m_pi_0
bp_d1p_e_pipi_basf2_kin_conf.m_4 = m_pi_0
bp_d1p_e_pipi_basf2_kin_conf.m_l = m_e
bp_d1p_e_pipi_basf2_kin_conf.m_nom = m_D1p_0
bp_d1p_e_pipi_basf2_kin_conf.g_nom = gam_D1p_0
bp_d1p_e_pipi_basf2_kin_conf.l = 0
bp_d1p_e_pipi_basf2_kin_conf.threebody = True

bp_d1p_mu_pipi_basf2_kin_conf = deepcopy(bp_d1p_e_pipi_basf2_kin_conf)
bp_d1p_mu_pipi_basf2_kin_conf.m_l = m_mu
bp_d1p_tau_pipi_basf2_kin_conf = deepcopy(bp_d1p_e_pipi_basf2_kin_conf)
bp_d1p_tau_pipi_basf2_kin_conf.m_l = m_tau


b0_d1p_e_pipi_basf2_kin_conf = kin_conf()
b0_d1p_e_pipi_basf2_kin_conf.m_1 = m_B_0
b0_d1p_e_pipi_basf2_kin_conf.m_2 = m_D_0
b0_d1p_e_pipi_basf2_kin_conf.m_3 = m_pi_pm
b0_d1p_e_pipi_basf2_kin_conf.m_4 = m_pi_0
b0_d1p_e_pipi_basf2_kin_conf.m_l = m_e
b0_d1p_e_pipi_basf2_kin_conf.m_nom = m_D1p_pm
b0_d1p_e_pipi_basf2_kin_conf.g_nom = gam_D1p_pm
b0_d1p_e_pipi_basf2_kin_conf.l = 0
b0_d1p_e_pipi_basf2_kin_conf.threebody = True

b0_d1p_mu_pipi_basf2_kin_conf = deepcopy(b0_d1p_e_pipi_basf2_kin_conf)
b0_d1p_mu_pipi_basf2_kin_conf.m_l = m_mu
b0_d1p_tau_pipi_basf2_kin_conf = deepcopy(b0_d1p_e_pipi_basf2_kin_conf)
b0_d1p_tau_pipi_basf2_kin_conf.m_l = m_tau


bp_d1p_e_pipi_2_basf2_kin_conf = kin_conf()
bp_d1p_e_pipi_2_basf2_kin_conf.m_1 = m_B_pm
bp_d1p_e_pipi_2_basf2_kin_conf.m_2 = m_Dstar_0 - 1e-5 # avoid numerical issues
bp_d1p_e_pipi_2_basf2_kin_conf.m_3 = m_pi_0
bp_d1p_e_pipi_2_basf2_kin_conf.m_4 = m_pi_0
bp_d1p_e_pipi_2_basf2_kin_conf.m_l = m_e
bp_d1p_e_pipi_2_basf2_kin_conf.m_nom = m_D1p_0
bp_d1p_e_pipi_2_basf2_kin_conf.g_nom = gam_D1p_0
bp_d1p_e_pipi_2_basf2_kin_conf.l = 0
bp_d1p_e_pipi_2_basf2_kin_conf.threebody = True

bp_d1p_mu_pipi_2_basf2_kin_conf = deepcopy(bp_d1p_e_pipi_2_basf2_kin_conf)
bp_d1p_mu_pipi_2_basf2_kin_conf.m_l = m_mu
bp_d1p_tau_pipi_2_basf2_kin_conf = deepcopy(bp_d1p_e_pipi_2_basf2_kin_conf)
bp_d1p_tau_pipi_2_basf2_kin_conf.m_l = m_tau


b0_d1p_e_pipi_2_basf2_kin_conf = kin_conf()
b0_d1p_e_pipi_2_basf2_kin_conf.m_1 = m_B_0
b0_d1p_e_pipi_2_basf2_kin_conf.m_2 = m_Dstar_pm - 1e-5 # avoid numerical issues
b0_d1p_e_pipi_2_basf2_kin_conf.m_3 = m_pi_0
b0_d1p_e_pipi_2_basf2_kin_conf.m_4 = m_pi_0
b0_d1p_e_pipi_2_basf2_kin_conf.m_l = m_e
b0_d1p_e_pipi_2_basf2_kin_conf.m_nom = m_D1p_pm
b0_d1p_e_pipi_2_basf2_kin_conf.g_nom = gam_D1p_pm
b0_d1p_e_pipi_2_basf2_kin_conf.l = 0
b0_d1p_e_pipi_2_basf2_kin_conf.threebody = True

b0_d1p_mu_pipi_2_basf2_kin_conf = deepcopy(b0_d1p_e_pipi_2_basf2_kin_conf)
b0_d1p_mu_pipi_2_basf2_kin_conf.m_l = m_mu
b0_d1p_tau_pipi_2_basf2_kin_conf = deepcopy(b0_d1p_e_pipi_2_basf2_kin_conf)
b0_d1p_tau_pipi_2_basf2_kin_conf.m_l = m_tau


bp_d0_e_pipi_basf2_kin_conf = kin_conf()
bp_d0_e_pipi_basf2_kin_conf.m_1 = m_B_pm
bp_d0_e_pipi_basf2_kin_conf.m_2 = m_Dstar_0 - 1e-5 # avoid numerical issues
bp_d0_e_pipi_basf2_kin_conf.m_3 = m_pi_0
bp_d0_e_pipi_basf2_kin_conf.m_4 = m_pi_0
bp_d0_e_pipi_basf2_kin_conf.m_l = m_e
bp_d0_e_pipi_basf2_kin_conf.m_nom = m_D0_0
bp_d0_e_pipi_basf2_kin_conf.g_nom = gam_D0_0
bp_d0_e_pipi_basf2_kin_conf.l = 0
bp_d0_e_pipi_basf2_kin_conf.threebody = True

bp_d0_mu_pipi_basf2_kin_conf = deepcopy(bp_d0_e_pipi_basf2_kin_conf)
bp_d0_mu_pipi_basf2_kin_conf.m_l = m_mu
bp_d0_tau_pipi_basf2_kin_conf = deepcopy(bp_d0_e_pipi_basf2_kin_conf)
bp_d0_tau_pipi_basf2_kin_conf.m_l = m_tau


b0_d0_e_pipi_basf2_kin_conf = kin_conf()
b0_d0_e_pipi_basf2_kin_conf.m_1 = m_B_0
b0_d0_e_pipi_basf2_kin_conf.m_2 = m_Dstar_pm - 1e-5  # avoid numerical issues
b0_d0_e_pipi_basf2_kin_conf.m_3 = m_pi_0
b0_d0_e_pipi_basf2_kin_conf.m_4 = m_pi_0
b0_d0_e_pipi_basf2_kin_conf.m_l = m_e
b0_d0_e_pipi_basf2_kin_conf.m_nom = m_D0_pm
b0_d0_e_pipi_basf2_kin_conf.g_nom = gam_D0_pm
b0_d0_e_pipi_basf2_kin_conf.l = 0
b0_d0_e_pipi_basf2_kin_conf.threebody = True

b0_d0_mu_pipi_basf2_kin_conf = deepcopy(b0_d0_e_pipi_basf2_kin_conf)
b0_d0_mu_pipi_basf2_kin_conf.m_l = m_mu
b0_d0_tau_pipi_basf2_kin_conf = deepcopy(b0_d0_e_pipi_basf2_kin_conf)
b0_d0_tau_pipi_basf2_kin_conf.m_l = m_tau


bp_d0_e_pipi_2_basf2_kin_conf = kin_conf()
bp_d0_e_pipi_2_basf2_kin_conf.m_1 = m_B_pm
bp_d0_e_pipi_2_basf2_kin_conf.m_2 = m_D_0
bp_d0_e_pipi_2_basf2_kin_conf.m_3 = m_pi_0
bp_d0_e_pipi_2_basf2_kin_conf.m_4 = m_pi_0
bp_d0_e_pipi_2_basf2_kin_conf.m_l = m_e
bp_d0_e_pipi_2_basf2_kin_conf.m_nom = m_D0_0
bp_d0_e_pipi_2_basf2_kin_conf.g_nom = gam_D0_0
bp_d0_e_pipi_2_basf2_kin_conf.l = 0
bp_d0_e_pipi_2_basf2_kin_conf.threebody = True

bp_d0_mu_pipi_2_basf2_kin_conf = deepcopy(bp_d0_e_pipi_2_basf2_kin_conf)
bp_d0_mu_pipi_2_basf2_kin_conf.m_l = m_mu
bp_d0_tau_pipi_2_basf2_kin_conf = deepcopy(bp_d0_e_pipi_2_basf2_kin_conf)
bp_d0_tau_pipi_2_basf2_kin_conf.m_l = m_tau


b0_d0_e_pipi_2_basf2_kin_conf = kin_conf()
b0_d0_e_pipi_2_basf2_kin_conf.m_1 = m_B_0
b0_d0_e_pipi_2_basf2_kin_conf.m_2 = m_D_0
b0_d0_e_pipi_2_basf2_kin_conf.m_3 = m_pi_pm
b0_d0_e_pipi_2_basf2_kin_conf.m_4 = m_pi_0
b0_d0_e_pipi_2_basf2_kin_conf.m_l = m_e
b0_d0_e_pipi_2_basf2_kin_conf.m_nom = m_D0_pm
b0_d0_e_pipi_2_basf2_kin_conf.g_nom = gam_D0_pm
b0_d0_e_pipi_2_basf2_kin_conf.l = 0
b0_d0_e_pipi_2_basf2_kin_conf.threebody = True

b0_d0_mu_pipi_2_basf2_kin_conf = deepcopy(b0_d0_e_pipi_2_basf2_kin_conf)
b0_d0_mu_pipi_2_basf2_kin_conf.m_l = m_mu
b0_d0_tau_pipi_2_basf2_kin_conf = deepcopy(b0_d0_e_pipi_2_basf2_kin_conf)
b0_d0_tau_pipi_2_basf2_kin_conf.m_l = m_tau


kin_conf_dict = {}
kin_conf_dict[('bp', 'd0', 'e')] = bp_d0_e_basf2_kin_conf
kin_conf_dict[('bp', 'd0', 'mu')] = bp_d0_mu_basf2_kin_conf
kin_conf_dict[('bp', 'd0', 'tau')] = bp_d0_tau_basf2_kin_conf
kin_conf_dict[('b0', 'd0', 'e')] = b0_d0_e_basf2_kin_conf
kin_conf_dict[('b0', 'd0', 'mu')] = b0_d0_mu_basf2_kin_conf
kin_conf_dict[('b0', 'd0', 'tau')] = b0_d0_tau_basf2_kin_conf
kin_conf_dict[('bp', 'd1p', 'e')] = bp_d1p_e_basf2_kin_conf
kin_conf_dict[('bp', 'd1p', 'mu')] = bp_d1p_mu_basf2_kin_conf
kin_conf_dict[('bp', 'd1p', 'tau')] = bp_d1p_tau_basf2_kin_conf
kin_conf_dict[('b0', 'd1p', 'e')] = b0_d1p_e_basf2_kin_conf
kin_conf_dict[('b0', 'd1p', 'mu')] = b0_d1p_mu_basf2_kin_conf
kin_conf_dict[('b0', 'd1p', 'tau')] = b0_d1p_tau_basf2_kin_conf
kin_conf_dict[('bp', 'd0', 'e', 'eta')] = bp_d0_e_eta_basf2_kin_conf
kin_conf_dict[('bp', 'd0', 'mu', 'eta')] = bp_d0_mu_eta_basf2_kin_conf
kin_conf_dict[('bp', 'd0', 'tau', 'eta')] = bp_d0_tau_eta_basf2_kin_conf
kin_conf_dict[('b0', 'd0', 'e', 'eta')] = b0_d0_e_eta_basf2_kin_conf
kin_conf_dict[('b0', 'd0', 'mu', 'eta')] = b0_d0_mu_eta_basf2_kin_conf
kin_conf_dict[('b0', 'd0', 'tau', 'eta')] = b0_d0_tau_eta_basf2_kin_conf
kin_conf_dict[('bp', 'd1p', 'e', 'eta')] = bp_d1p_e_eta_basf2_kin_conf
kin_conf_dict[('bp', 'd1p', 'mu', 'eta')] = bp_d1p_mu_eta_basf2_kin_conf
kin_conf_dict[('bp', 'd1p', 'tau', 'eta')] = bp_d1p_tau_eta_basf2_kin_conf
kin_conf_dict[('b0', 'd1p', 'e', 'eta')] = b0_d1p_e_eta_basf2_kin_conf
kin_conf_dict[('b0', 'd1p', 'mu', 'eta')] = b0_d1p_mu_eta_basf2_kin_conf
kin_conf_dict[('b0', 'd1p', 'tau', 'eta')] = b0_d1p_tau_eta_basf2_kin_conf
kin_conf_dict[('bp', 'd0', 'e', 'dstar_pipi')] = bp_d0_e_pipi_basf2_kin_conf
kin_conf_dict[('bp', 'd0', 'mu', 'dstar_pipi')] = bp_d0_mu_pipi_basf2_kin_conf
kin_conf_dict[('bp', 'd0', 'tau', 'dstar_pipi')] = bp_d0_tau_pipi_basf2_kin_conf
kin_conf_dict[('b0', 'd0', 'e', 'dstar_pipi')] = b0_d0_e_pipi_basf2_kin_conf
kin_conf_dict[('b0', 'd0', 'mu', 'dstar_pipi')] = b0_d0_mu_pipi_basf2_kin_conf
kin_conf_dict[('b0', 'd0', 'tau', 'dstar_pipi')] = b0_d0_tau_pipi_basf2_kin_conf
kin_conf_dict[('bp', 'd0', 'e', 'd_pipi')] = bp_d0_e_pipi_2_basf2_kin_conf
kin_conf_dict[('bp', 'd0', 'mu', 'd_pipi')] = bp_d0_mu_pipi_2_basf2_kin_conf
kin_conf_dict[('bp', 'd0', 'tau', 'd_pipi')] = bp_d0_tau_pipi_2_basf2_kin_conf
kin_conf_dict[('b0', 'd0', 'e', 'd_pipi')] = b0_d0_e_pipi_2_basf2_kin_conf
kin_conf_dict[('b0', 'd0', 'mu', 'd_pipi')] = b0_d0_mu_pipi_2_basf2_kin_conf
kin_conf_dict[('b0', 'd0', 'tau', 'd_pipi')] = b0_d0_tau_pipi_2_basf2_kin_conf
kin_conf_dict[('bp', 'd1p', 'e', 'd_pipi')] = bp_d1p_e_pipi_basf2_kin_conf
kin_conf_dict[('bp', 'd1p', 'mu', 'd_pipi')] = bp_d1p_mu_pipi_basf2_kin_conf
kin_conf_dict[('bp', 'd1p', 'tau', 'd_pipi')] = bp_d1p_tau_pipi_basf2_kin_conf
kin_conf_dict[('b0', 'd1p', 'e', 'd_pipi')] = b0_d1p_e_pipi_basf2_kin_conf
kin_conf_dict[('b0', 'd1p', 'mu', 'd_pipi')] = b0_d1p_mu_pipi_basf2_kin_conf
kin_conf_dict[('b0', 'd1p', 'tau', 'd_pipi')] = b0_d1p_tau_pipi_basf2_kin_conf
kin_conf_dict[('bp', 'd1p', 'e', 'dstar_pipi')] = bp_d1p_e_pipi_2_basf2_kin_conf
kin_conf_dict[('bp', 'd1p', 'mu', 'dstar_pipi')] = bp_d1p_mu_pipi_2_basf2_kin_conf
kin_conf_dict[('bp', 'd1p', 'tau', 'dstar_pipi')] = bp_d1p_tau_pipi_2_basf2_kin_conf
kin_conf_dict[('b0', 'd1p', 'e', 'dstar_pipi')] = b0_d1p_e_pipi_2_basf2_kin_conf
kin_conf_dict[('b0', 'd1p', 'mu', 'dstar_pipi')] = b0_d1p_mu_pipi_2_basf2_kin_conf
kin_conf_dict[('b0', 'd1p', 'tau', 'dstar_pipi')] = b0_d1p_tau_pipi_2_basf2_kin_conf


# Here we add the rho mu nu decay
rho_kin_conf = kin_conf()
rho_kin_conf.m_1 = 5279.34e-3
rho_kin_conf.m_2 = 139.57039e-3
rho_kin_conf.m_3 = 139.57039e-3
rho_kin_conf.m_l = 105.65837e-3
rho_kin_conf.m_nom = 775.26e-3
rho_kin_conf.g_nom = 147.8e-3
rho_kin_conf.l = 1

rho_bcl_conf = bcl_conf()
rho_bcl_conf.params_a0 = [8 * 0.291 * rho_kin_conf.m_1 * rho_kin_conf.m_nom / (rho_kin_conf.m_1**2 - rho_kin_conf.m_nom**2), -0.861, 1.444]
rho_bcl_conf.params_a1 = [0.266, 0.378, 0.165]
rho_bcl_conf.params_a12 = [0.291, 0.718, 0.384]
rho_bcl_conf.params_v = [0.331, -0.876, 1.907]

ff_conf_dict[('rho')] = rho_bcl_conf
kin_conf_dict[('bp', 'rho', 'mu')] = rho_kin_conf
