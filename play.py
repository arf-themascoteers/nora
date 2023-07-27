import numpy as np
import pandas as pd
s = pd.Series([2, 3, 4, 5], name='f1',
              index=pd.Index(['p', 'q', 'r', 's'], name='idx'))

#print(s)
s = s.reset_index(name="values")
print(s)