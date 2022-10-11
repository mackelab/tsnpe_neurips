import pickle
from multicompartment.models import setup_l5pc, simulate_l5pc, summstats_l5pc
from multicompartment.models.l5pc.utils import return_gt

gt = return_gt()
setup_l5pc(load_libraries=False)
trace = simulate_l5pc(gt)
xo = summstats_l5pc(trace)

with open("xo_trace.pkl", "wb") as handle:
    pickle.dump(trace, handle)
xo.to_pickle("xo.pkl")
