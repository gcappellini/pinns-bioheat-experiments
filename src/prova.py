import utils as uu
import os
import coeff_calc as cc


current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
git_dir = os.path.dirname(src_dir)
tests_dir = os.path.join(git_dir, "tests")
os.makedirs(tests_dir, exist_ok=True)

df = uu.load_from_pickle(f"{src_dir}/data/vessel/cooling_meas_1.pkl")

y1_0 = uu.scale_t(22.28)
y2_0 = uu.scale_t(26.67)
y3_0 = uu.scale_t(21.5)

print(y1_0, y2_0, y3_0, cc.a5)


