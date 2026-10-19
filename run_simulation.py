"""End-to-end run: data -> Elo/features -> XGBoost -> score layer -> N tournament simulations."""
import argparse, json
import pandas as pd
from afc_sim import MatchEngine, ShootoutModel, Simulator, build_features, load_raw, train_xgb

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data"); ap.add_argument("--sims", type=int, default=10_000)
ap.add_argument("--seed", type=int, default=2027); ap.add_argument("--train-through", type=int, default=2026)
ap.add_argument("--et-factor", type=float, default=0.8); ap.add_argument("--out", default="simulation_results.csv")
a = ap.parse_args()

res, sho = load_raw(a.data)
df, state = build_features(res)
model = train_xgb(df, a.train_through)
engine = MatchEngine(model, state)
shoot = ShootoutModel.fit(df, sho)
print("shoot-out model:", json.dumps(shoot.diagnostics, indent=1))
out = Simulator(engine, shoot, et_factor=a.et_factor).run(a.sims, a.seed)
pd.set_option("display.width", 200)
print(out.table.round(3).to_string())
print(json.dumps(out.diagnostics, indent=1, default=float))
out.table.round(5).to_csv(a.out)
