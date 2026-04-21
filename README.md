# MCAmbulance
MCAmbulance is a small python package to correct Monte-Carlo samples of $B \rightarrow D_0^\ast \ell \nu$ and $B \rightarrow D_1^\prime \ell \nu$ decays produced by the [basf2](https://github.com/belle2/basf2) version of [EvtGen](https://evtgen.hepforge.org) for missing phase-space factors.

MCAmbulance relies on
- numpy
- scipy

The missing phase-space factors and the reweighting strategy are described in detail in the preprint [arXiv:2602.18378](https://arxiv.org/abs/2602.18378).
MCAmbulance uses masses and widths based on the default values in [basf2](https://github.com/belle2/basf2), taken from [evt.pdl](https://github.com/belle2/basf2/blob/main/framework/particledb/data/evt.pdl).

The following decays are supported:
- $B \rightarrow D_0^\ast(\rightarrow D \pi) \ell \nu$, based on [DECAY_BELLE2.DEC](https://github.com/belle2/basf2/blob/main/decfiles/dec/DECAY_BELLE2.DEC)
- $B \rightarrow D_0^\ast(\rightarrow D \eta) \ell \nu$, based on [1196708001.dec](https://github.com/belle2/basf2/blob/main/decfiles/dec/1196708001.dec) and [1296708001.dec](https://github.com/belle2/basf2/blob/main/decfiles/dec/1296708001.dec)
- $B \rightarrow D_0^\ast(\rightarrow D \pi \pi) \ell \nu$, based on [1196700003.dec](https://github.com/belle2/basf2/blob/main/decfiles/dec/1196700003.dec) and [1296700003.dec](https://github.com/belle2/basf2/blob/main/decfiles/dec/1296700003.dec) (Note, this decay violates parity)
- $B \rightarrow D_0^\ast(\rightarrow D^\ast \pi \pi) \ell \nu$, based on [1196700004.dec](https://github.com/belle2/basf2/blob/main/decfiles/dec/1196700004.dec) and [1296700004.dec](https://github.com/belle2/basf2/blob/main/decfiles/dec/1296700004.dec)
- $B \rightarrow D_1^\prime(\rightarrow D^\ast \pi) \ell \nu$, based on [DECAY_BELLE2.DEC](https://github.com/belle2/basf2/blob/main/decfiles/dec/DECAY_BELLE2.DEC)
- $B \rightarrow D_1^\prime(\rightarrow D^\ast \eta) \ell \nu$, based on [1196708000.dec](https://github.com/belle2/basf2/blob/main/decfiles/dec/1196708000.dec) and [1296708000.dec](https://github.com/belle2/basf2/blob/main/decfiles/dec/1296708000.dec)
- $B \rightarrow D_1^\prime(\rightarrow D \pi \pi) \ell \nu$, based on [1196700001.dec](https://github.com/belle2/basf2/blob/main/decfiles/dec/1196700001.dec) and [1296700001.dec](https://github.com/belle2/basf2/blob/main/decfiles/dec/1296700001.dec)
- $B \rightarrow D_1^\prime(\rightarrow D^\ast \pi \pi) \ell \nu$, based on [1196700002.dec](https://github.com/belle2/basf2/blob/main/decfiles/dec/1196700002.dec) and [1296700002.dec](https://github.com/belle2/basf2/blob/main/decfiles/dec/1296700002.dec)

In addition, we provide ```ISGW2``` implementations of
- $B \rightarrow D_0^\ast(\rightarrow D \pi) \ell \nu$
- $B \rightarrow D_1^\prime(\rightarrow D^\ast \pi) \ell \nu$

Note that decays simulated with the ```PHSP``` model are not affected, and so we do not provide code to reweight them.

# Installation
1. Clone this repository into a local directory.
```bash
   git clone <repo-url>
   ```
2. Add the repository root (the directory that contains the `mcambulance/` folder) to your `PYTHONPATH`:
```bash
   export PYTHONPATH="/path/to/MCAmbulance:${PYTHONPATH}"
   ```
### Alternative (script/notebook only)
If you don't want to set environment variables, you can add the repo to `sys.path` at runtime:
```python
import sys
sys.path.insert(0, "/path/to/MCAmbulance")  # repo root containing `mcambulance/`
import mcambulance
```

# Usage
Correcting $B^\pm \rightarrow D_0^\ast(\rightarrow D \pi) \mu \nu_\mu$ samples is achieved through:
```
from mcambulance import MCAmbulance

d0_d_pi_fix = MCAmbulance('bp', 'd0_d_pi', 'mu')

weights = d0_d_pi_fix.CorrectionWeight(dpi_invM)
```

Here, ```dpi_invM``` is a numpy array containing the **truth-level** $D\pi$ invariant mass for each affected event.
The numpy array ```weights``` now contains the correction weights for each event.

The constructor of ```MCAmbulance``` takes three arguments:
- ```bmeson```: ```bp``` or ```b0``` for charged or neutral $B$ mesons, respectively
- ```channel```: ```d0_d_pi```, ```d0_d_eta```, ```d0_d_pipi```, ```d0_dstar_pipi```, ```d1p_dstar_pi```, ...
- ```lepton```: ```e```, ```mu``` or ```tau```

To select the ```ISGW2``` implementations, ```channel``` is either ```d0_d_pi_isgw2``` or ```d1p_dstar_pi_isgw2```.

In addition, we support the ```BCL``` model for $B^+\rightarrow\rho^0\mu^+\nu_\mu$ decays.
This EvtGen model has an additional problem, a division by zero in a basis change, which we reproduce as closely as feasible.

# References
If you use MCAmbulance, you should cite the following references:
- MCAmbulance: Florian Herren and Raynette van Tonder, [arXiv:2602.18378](https://arxiv.org/abs/2602.18378)
- [EvtGen](https://evtgen.hepforge.org): David J. Lange, [Nucl.Instrum.Meth.A 462 (2001) 152-155](https://doi.org/10.1016/S0168-9002(01)00089-4)
- [basf2](https://github.com/belle2/basf2): Belle-II Framework Software Group, [Comput.Softw.Big Sci. 3 (2019) 1, 1](https://doi.org/10.1007/s41781-018-0017-9)
- LLSW parametrisation: Adam K. Leibovich, Zoltan Ligeti, Ian M. Stewart and Mark B. Wise, [Phys.Rev.Lett. 78 (1997) 3995-3998](https://doi.org/10.1103/PhysRevLett.78.3995) & [Phys.Rev.D 57 (1998) 308-330](https://doi.org/10.1103/PhysRevD.57.308)
- Default form factor values: Florian U. Bernlochner and Zoltan Ligeti, [Phys.Rev.D 95 (2017) 1, 014022](https://doi.org/10.1103/PhysRevD.95.014022)
- If using the ISGW2 form factors: Daryl Scora and Nathan Isgur, [Phys.Rev.D 52 (1992) 2783-2812](https://doi.org/10.1103/PhysRevD.52.2783)

# Authors
 * Florian Herren <florian.s.herren@gmail.com>
 * Raynette van Tonder <raynette.vantonder@kit.edu>

# Contributors
 * Ilias Tsaklidis
