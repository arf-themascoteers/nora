import re


def get_band_columns(columns):
    return [x for x in columns if x.startswith("B") and len(re.findall(r'\d+', x)) == 1]


if __name__ == "__main__":
    cols = "row	column	scene	elevation	som	moisture	temp	B01	B02	B03	B04	B05	B06	B07	B8A	B09	B11	B12"
    cols = cols.split()
    print(get_band_columns(cols))