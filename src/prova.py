import numpy as np
import matlab.engine
import os
import plots as pp
import common as co
import utils as uu

current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)

prj = "change_matlab"
run="perf_muscle"
co.set_prj(prj)
run_figs = co.set_run(run)

eng = matlab.engine.start_matlab()
eng.cd(src_dir, nargout=0)
eng.BioHeat(nargout=0)
eng.quit()

t, weights = uu.load_weights()
mus = uu.compute_mu()
pp.plot_weights(weights, t, run_figs, gt=True)
pp.plot_mu(mus, t, run_figs, gt=True)