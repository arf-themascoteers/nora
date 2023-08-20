import rasterio


path = r"D:\Data\Tim\Created\Vectis\Higher\SKYWATCH_PHR_PS_20220204T0037_RGBN_Tile_0_0_e031.tif"

with rasterio.open(path) as src:
    print(src.count)
