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

import numpy as np
from .misc import lam
from .semileptonic import BtoDstarstarlnu

class BtoRholnu_BCL(BtoDstarstarlnu):
    def __init__(self, kin_conf, bcl_conf):
        self.params_a0 = bcl_conf.params_a0
        self.params_a1 = bcl_conf.params_a1
        self.params_a12 = bcl_conf.params_a12
        self.params_v = bcl_conf.params_v
        BtoDstarstarlnu.__init__(self, kin_conf)

    def _helamps(self, q2, M2):
        V, A0, A1, A2 = self._ffs(q2, M2)
        
        M = np.sqrt(M2)
        sqrtlam = np.sqrt(lam(M2, q2, self.m_1**2))
        sqrtq2 = np.sqrt(q2)

        h0 = 8. * self.m_1 * M / sqrtq2 * (- A2 * sqrtlam ** 2 + (self.m_1 + M) ** 2 * (self.m_1 ** 2 - M2 - q2) * A1) / 16. / self.m_1 / M2 / (self.m_1 + M)
        ht = A0 * sqrtlam / sqrtq2
        hp = (A1 * (self.m_1 + M) ** 2 + sqrtlam * V) / (self.m_1 + M)
        hm = (-A1 * (self.m_1 + M) ** 2 + sqrtlam * V) / (self.m_1 + M)
                          
        return h0, ht, hp, hm

    def _ffs(self, q2, M2):
        # The EvtGen module uses M_rho to define z, this is incorrect but likely of not much consequence
        tplus = (self.m_1 + self.m_nom) ** 2
        tminus = (self.m_1 - self.m_nom) ** 2
        tzero = tplus * (1. - np.sqrt(1. - tminus/tplus))

        # Resonance masses
        mR2A0 = 5.279**2
        mR2V = 5.325**2
        # Technically, the axial vector is above threshold and should not be included
        mR2A1 = 5.724**2
        mR2A12 = mR2A1

        poleA0 = 1. / (1. - q2 / mR2A0)
        poleA1 = 1. / (1. - q2 / mR2A1)
        poleA12 = 1. / (1. - q2 / mR2A12)
        poleV = 1. / (1. - q2 / mR2V)
        z = (np.sqrt(tplus - q2) - np.sqrt(tplus - tzero)) / (np.sqrt(tplus - q2) + np.sqrt(tplus - tzero))
        z0 = (np.sqrt(tplus) - np.sqrt(tplus - tzero)) / (np.sqrt(tplus) + np.sqrt(tplus - tzero))
        zz = z - z0 

        A0 = poleA0 * (self.params_a0[0] + self.params_a0[1] * zz + self.params_a0[2] * zz * zz)
        A1 = poleA1 * (self.params_a1[0] + self.params_a1[1] * zz + self.params_a1[2] * zz * zz)
        A12 = poleA12 * (self.params_a12[0] + self.params_a12[1] * zz + self.params_a12[2] * zz * zz)
        V = poleV * (self.params_v[0] + self.params_v[1] * zz + self.params_v[2] * zz * zz)

        # The EvtGen module divides by 0 since it performs the basis change from (A1, A12) -> (A1, A2) not with the invariant mass, but the nominal resonance mass
        # This leads to a spike at q2 = (MB - Mrho)^2, which we need to regulate in our code
        # Note, this still will lead to warnings when integrating over q2, for example when determining the normalization
        # Ideally we should adopt a better extrapolation to beyond the narrow width limit...
        if np.abs(q2 - tminus) > 1e-2:
            kaellen = (tplus - q2) * (tminus - q2)
        else:
            if tminus > q2:
                kaellen = (tplus - q2) * 1e-2
            else:
                kaellen = -(tplus - q2) * 1e-2
        A2 = (tplus * (self.m_1 ** 2 - self.m_nom ** 2 - q2) * A1 - 16. * self.m_1 * self.m_nom ** 2 * (self.m_1 + self.m_nom) * A12) / kaellen

        return V, A0, A1, A2
