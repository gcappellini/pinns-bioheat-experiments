# PINNs BIOHEAT EXPERIMENTS
In this project we are analyzing the experiments conducted at the AMC Radiotherapy Department for Hyperthermia Treatment Planning (HTP). 

The experiments can be subdivided in three categories:

1. **Agar**: Experiments conducted in the framework of another project of a thesist. The data can be used as a preliminary test of our NBHO. 
    Implementation edits: only 6 measuring points, better plots.
    Theory: mm-observer to retrieve the thermal conductivity and, eventually, the convection coefficient

2. **Phantom**: First experiments conducted with the RF antenna and the water bolus on a phantom
    Implementation: a df for storing the measurements + transformation that maps from df to actual position in the phantom
    Theory: asses power with superposition w.r.t. Dolphinx/Mathematica; mm-observer to retrieve the thermal conductivity and, eventually, the convection coefficient

3. **Vessel**: The previous phantom is now prefused by making water flow through one of the catheters, at different speeds
    Implementation:
    Theory: verify the power, compute the velocity, MEASURE THE HEIGHT of the bucket

