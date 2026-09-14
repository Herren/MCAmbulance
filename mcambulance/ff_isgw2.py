#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Based on the ISGW2 implementation within EvtGen:
# Copyright (c) 1998-2020 CERN for the benefit of the EvtGen authors
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

def Getas(m, lqcd2):
    flav = 4
    als = 0.6
    if m > 0.6:
        if m < 1.85:
            nflav = 3
        als = 12. * np.pi / (33. - 2. * nflav) / np.log(m ** 2 / lqcd2)
    return als

class BtoD0lnu_ISGW2(BtoDstarstarlnu):
    def __init__(self, kin_conf, isgw2_conf):
        self.lqcd2 = isgw2_conf.lqcd2
        self.nfp = isgw2_conf.nfp
        self.mqm = isgw2_conf.mqm
        self.msb = isgw2_conf.msb
        self.msd = isgw2_conf.msd
        self.bb2 = isgw2_conf.bb2
        self.mbb = isgw2_conf.mbb
        self.msq = isgw2_conf.msq
        self.bx2 = isgw2_conf.bx2
        self.mbx = isgw2_conf.mbx
        self.mtb = self.msb + self.msd
        self.mtx = self.msq + self.msd
        self.bbx2 = 0.5 * ( self.bb2 + self.bx2 )
        BtoDstarstarlnu.__init__(self, kin_conf)
        
    def _helamps(self, q2, M2):
        fp, fm = self._ffs(q2, M2)
        
        M = np.sqrt(M2)
        sqrtlam = np.sqrt(lam(M2, q2, self.m_1**2))
        sqrtq2 = np.sqrt(q2)

        f0 = (fp + ( fm / ( ( self.m_1**2 - M2 ) / q2 ) ))

        h0 = sqrtlam * fp / sqrtq2
        ht = (self.m_1 ** 2 - M2) * f0 / sqrtq2
        hp = 0.
        hm = 0.
                          
        return h0, ht, hp, hm

    def _ffs(self, q2, M2):
        mb = self.m_1
        mx = np.sqrt(M2)
        tm = ( mb - mx ) ** 2
        t = q2 * np.heaviside(tm - q2, 1.) + 0.99 * tm * np.heaviside(q2 - tm, 0.)

        r2 = 3.0 / ( 4.0 * self.msb * self.msq ) + 3 * self.msd ** 2 / ( 2 * self.mbb * self.mbx * self.bbx2 ) + ( 16.0 / ( self.mbb * self.mbx * ( 33.0 - 2.0 * self.nfp ) ) ) * np.log( Getas( self.mqm, self.lqcd2 ) / Getas( self.msq, self.lqcd2 ) )
        f5 = np.sqrt( self.mtx / self.mtb ) * ( np.sqrt( self.bx2 * self.bb2 ) / self.bbx2) ** 2.5 / ( ( 1.0 + r2 * ( tm - t ) / 18.0 ) ** 3.0 )
        f5uppum = f5 / np.sqrt( self.mbb / self.mtb ) * np.sqrt( self.mbx / self.mtx )
        f5upmum = f5 * np.sqrt( self.mbb / self.mtb ) / np.sqrt( self.mbx / self.mtx )

        uppum = -1.0 * f5uppum * np.sqrt( 2.0 / ( 3.0 * self.bb2 ) ) * self.msd
        upmum = 1.0 * f5upmum * np.sqrt( 2.0 / ( 3.0 * self.bb2 ) ) * self.msd * self.mtb / self.mtx
        return ( uppum + upmum ) / 2.0, ( uppum - upmum ) / 2.0

class BtoD1plnu_ISGW2(BtoDstarstarlnu):
    def __init__(self, kin_conf, isgw2_conf):
        self.lqcd2 = isgw2_conf.lqcd2
        self.nfp = isgw2_conf.nfp
        self.mqm = isgw2_conf.mqm
        self.msb = isgw2_conf.msb
        self.msd = isgw2_conf.msd
        self.bb2 = isgw2_conf.bb2
        self.mbb = isgw2_conf.mbb
        self.msq = isgw2_conf.msq
        self.bx2 = isgw2_conf.bx2
        self.mbx = isgw2_conf.mbx
        self.mtb = self.msb + self.msd
        self.mtx = self.msq + self.msd
        self.bbx2 = 0.5 * ( self.bb2 + self.bx2 )
        self.mum = 1. / (1. / self.msq - 1. / self.msb)
        BtoDstarstarlnu.__init__(self, kin_conf)
        
    def _helamps(self, q2, M2):
        gf, ff, apf, amf = self._ffs(q2, M2)

        M = np.sqrt(M2)
        sqrtlam = np.sqrt(lam(M2,q2,self.m_1**2))
        sqrtq2 = np.sqrt(q2)
        
        V = gf * (self.m_1 + M)
        A1 = ff / (self.m_1 + M)
        A2 = -apf * (self.m_1 + M)
        a3f = ff / 2. / M + apf * (self.m_1 ** 2 - M2) / 2. / M
        A0 = a3f + q2 * amf / 2. / M

        h0 = (A1 * (self.m_1 + M) ** 2 * (self.m_1 ** 2 - M2 - q2) - A2 * sqrtlam ** 2) / (2. * M * (self.m_1 + M) * sqrtq2)
        ht = A0 * sqrtlam / sqrtq2
        hp = (A1 * (self.m_1 + M) ** 2 - sqrtlam * V) / (self.m_1 + M)
        hm = (A1 * (self.m_1 + M) ** 2 + sqrtlam * V) / (self.m_1 + M)
                          
        return h0, ht, hp, hm

    def _ffs(self, q2, M2):
        mb = self.m_1
        mx = np.sqrt(M2)
        tm = ( mb - mx ) ** 2
        t = q2 * np.heaviside(tm - q2, 1.) + 0.99 * tm * np.heaviside(q2 - tm, 0.)
        wt = 1. + (tm - t) / 2. / self.mbb / self.mbx

        r2 = 3.0 / ( 4.0 * self.msb * self.msq ) + 3. * self.msd ** 2 / ( 2. * self.mbb * self.mbx * self.bbx2 ) + ( 16.0 / ( self.mbb * self.mbx * ( 33.0 - 2.0 * self.nfp ) ) ) * np.log( Getas( self.mqm, self.lqcd2 ) / Getas( self.msq, self.lqcd2 ) )
        f5 = np.sqrt( self.mtx / self.mtb ) * ( np.sqrt( self.bx2 * self.bb2 ) / self.bbx2) ** 2.5 / ( ( 1.0 + r2 * ( tm - t ) / 18.0 ) ** 3.0 )
        f5q = f5 / np.sqrt(self.mbb / self.mtb) / np.sqrt(self.mbx / self.mtx)
        f5l = f5 * np.sqrt(self.mbb / self.mtb) * np.sqrt(self.mbx / self.mtx)
        f5cppcm = f5 / (np.sqrt(self.mbb / self.mtb) ** 3) * np.sqrt(self.mbx / self.mtx)
        f5cpmcm = f5q

        ql = f5q * np.sqrt(1. / 6.) * self.msd / (np.sqrt(self.bb2) * self.mtx) * (1.0 - self.bb2 * self.mtb / 4. / self.msd / self.msq / self.msb)
        ll = f5l * np.sqrt(2. / 3.) * self.mtb * np.sqrt(self.bb2) * (0.5 / self.msq - 1.5 / self.msb + self.msd * self.mtx * (wt - 1.) / self.bb2 * (1.0 / self.msq - self.msd * self.bb2 / 2. / self.mum / self.mtx / self.bbx2))
        cppcm = f5cppcm * self.msd ** 2 * self.bx2 / (np.sqrt(6. * self.bb2) * self.mtb * self.msq * self.bbx2)
        cpmcm = - np.sqrt(2. / 3.) * self.msd * f5cpmcm / self.mtx / np.sqrt(self.bb2) * (1. + self.msd * self.bx2 / 2. / self.msq / self.bbx2)

        # smearing present in evtgen
        parMass = self.m_1
        q2max = parMass*parMass + M2 - 2.*parMass*np.sqrt(M2)
        massNom = self.m_nom
        q2maxNom = parMass*parMass + massNom*massNom - 2.*parMass*massNom
        q2maxin = np.sqrt(q2maxNom / q2max)
        if q2maxin > 1000:
            q2maxin = 1000.
        ql *= q2maxin
        ll *= q2maxin
        cppcm *= q2maxin
        cpmcm *= q2maxin
        
        return ql, ll, 0.5 * (cppcm + cpmcm), 0.5 * (cppcm - cpmcm)
