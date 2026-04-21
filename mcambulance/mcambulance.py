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
from copy import deepcopy

from .ff_hqet import BtoD0lnu_HQET, BtoD1plnu_HQET
from .ff_isgw2 import BtoD0lnu_ISGW2, BtoD1plnu_ISGW2
from .ff_bcl import BtoRholnu_BCL
from .defaults import ff_conf_dict, kin_conf_dict

class MCAmbulance:
    def __init__(self, bmeson, channel, lepton):
        self._decay = None

        if channel == "d0_d_pi":
            self._decay = BtoD0lnu_HQET(kin_conf_dict[(bmeson, "d0", lepton)], ff_conf_dict["d0"])
        elif channel == "d1p_dstar_pi":
            self._decay = BtoD1plnu_HQET(kin_conf_dict[(bmeson, "d1p", lepton)], ff_conf_dict["d1p"])
        elif channel == "d0_d_eta":
            self._decay = BtoD0lnu_HQET(kin_conf_dict[(bmeson, "d0", lepton, "eta")], ff_conf_dict["d0"])
        elif channel == "d1p_dstar_eta":
            self._decay = BtoD1plnu_HQET(kin_conf_dict[(bmeson, "d1p", lepton, "eta")], ff_conf_dict["d1p"])
        elif channel == "d1p_d_pipi":
            self._decay = BtoD1plnu_HQET(kin_conf_dict[(bmeson, "d1p", lepton, "d_pipi")], ff_conf_dict["d1p"])
        elif channel == "d1p_dstar_pipi":
            self._decay = BtoD1plnu_HQET(kin_conf_dict[(bmeson, "d1p", lepton, "dstar_pipi")], ff_conf_dict["d1p"])
        elif channel == "d0_d_pipi":
            self._decay = BtoD0lnu_HQET(kin_conf_dict[(bmeson, "d0", lepton, "d_pipi")], ff_conf_dict["d0"])
        elif channel == "d0_dstar_pipi":
            self._decay = BtoD0lnu_HQET(kin_conf_dict[(bmeson, "d0", lepton, "dstar_pipi")], ff_conf_dict["d0"])
        elif channel == "d0_d_pi_isgw2":
            self._decay = BtoD0lnu_ISGW2(kin_conf_dict[(bmeson, "d0", lepton)], ff_conf_dict["d0_isgw2"])
        elif channel == "d1p_dstar_pi_isgw2":
            self._decay = BtoD1plnu_ISGW2(kin_conf_dict[(bmeson, "d1p", lepton)], ff_conf_dict["d1p_isgw2"])
        elif channel == "rho":
            print("Initializing the B+ -> rho mu nu code, this will generate warnings due to a singularity in the q2 spectrum present in the EvtGen module.")
            if lepton != "mu" or bmeson != "bp":
                raise Exception("Only B+ -> rho0 mu nu decays are supported")
            self._decay = BtoRholnu_BCL(kin_conf_dict[("bp", "rho", "mu")], ff_conf_dict["rho"])
        else:
            raise Exception("Decay not supported")

    def CorrectionWeight(self, M):
        return self._decay.CorrectionWeight(M)
