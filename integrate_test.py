import pandas as pd

dest1 = "data/dest1.csv"
dest2 = "data/dest2.csv"
c1 = "data/processed/S2B_MSIL2A_20220413T002709_N0400_R016_T54HXE_20220413T021511/complete.csv"
c2 = "data/hi1p/complete.csv"

c1_df = pd.read_csv(c1)
c1_df.sort_values(["lon","lat","when"], inplace=True)
c1_df.reset_index(inplace=True)
c2_df = pd.read_csv(c2)
c2_df.sort_values(["lon","lat","when"], inplace=True)
c2_df.reset_index(inplace=True)

c3 = pd.concat([c1_df,c2_df], axis=1)
c3.to_csv(dest1, index=False)

c2_df = c2_df[["B04","B03","B02","B08"]]
c2_df.columns = ["red","green","blue","nir"]

c3 = pd.concat([c1_df,c2_df], axis=1)
c3.to_csv(dest2, index=False)