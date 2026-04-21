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

class kin_conf:
    def __init__(self):
        self.m_1 = 0.
        self.m_2 = 0.
        self.m_3 = 0.
        self.m_4 = 0.
        self.m_l = 0.
        self.m_nom = 0.
        self.g_nom = 0.
        self.RBW = 3.
        self.l = 0
        self.subthreshold = False
        self.threebody = False
        
class hqet_conf:
    def __init__(self):   
        self.params = []
        self.unc = None
        self.corr = None
        self.m_b = 0.
        self.m_c = 0.
        self.tau_w1 = 0.
        self.tau_wp = 0.
        self.L = 0.
        self.Lps = 0.
        self.zt_1 = 0
        self.zt_2 = 0.
        self.ce_1 = 0.
        self.ce_2 = 0.
        self.ce_3 = 0.
        self.ce_b = 0.

class isgw2_conf:
    def __init__(self):
        self.lqcd2 = 0.04
        self.nfp = 3.0
        self.mqm = 0.1
        self.msb = 5.2
        self.msd = 0.33
        self.bb2 = 0.431 * 0.431
        self.mbb = 5.31
        self.msq = 1.82
        self.bx2 = 0.33 * 0.33
        self.mbx = None

class bcl_conf:
    def __init__(self):
        self.params_a0 = []
        self.params_a1 = []
        self.params_a12 = []
        self.params_v = []
