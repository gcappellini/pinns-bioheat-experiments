import utils as uu
import os


current_file = os.path.abspath(__file__)
src_dir = os.path.dirname(current_file)
git_dir = os.path.dirname(src_dir)
tests_dir = os.path.join(git_dir, "tests")
os.makedirs(tests_dir, exist_ok=True)

df = uu.load_from_pickle(f"{src_dir}/data/vessel/cooling_meas_1.pkl")

print(df)


