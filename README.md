# dystonia_ebcc
Code to simulate mouse models of dystonia during EyeBlink Classical Conditioning (EBCC), described in Geminiani et al., *Front Syst Neurosci*, 2022.

Requirements:
- <a href="https://github.com/dbbs-lab/bsb">BSB</a> v3.8+
- NEST simulator v 2.18
- <a href="https://github.com/dbbs-lab/cereb-nest">cereb-nest</a>  NEST extension module

Repository content:
- `simulation` includes all the script to run simulations in control and pathological conditions. The main simulation file is `simulate_network.py`; different simulation configurations are described in the .json files
- `simulation_analysis.py` allows to plot figures from simulation results (which are saved as .hdf5 files) 
