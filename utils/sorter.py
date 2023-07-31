import re

# bands = "B01 B02	B03	B04	B05	B06	B07	B09	B11	B12	B8A"
# bands = bands.split()
# print(bands)
# bands = sorted(bands, key=lambda x: int(re.findall(r'\d+', x)[0]))
# print(bands)

folders = ["R10m","R60m","R20m"]
folders = sorted(folders, key=lambda x: int(re.findall(r'\d+', x)[0]))
print(folders)