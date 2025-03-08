import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

#regions
EUROPE = {'Europe' : [65, 35, -15, 30]}
W_ASIA = {'W_Asia' : [65, 35, 30, 80]}
E_ASIA = {'E_Asia' : [65, 35, 90, 140]}
W_AMERICA = {'W_America' : [65, 35, -140, -100]}
E_AMERICA = {'E_America' : [65, 35, -95, -55]}

regions = [EUROPE, W_ASIA, E_ASIA, W_AMERICA, E_AMERICA]
for region in regions:
    region_name, [lat_north, lat_south, lon_west, lon_east] = region.popitem()
    lon_west_A = lon_west - 20
    lon_east_A = lon_east + 20
    #create a map with PlateCarree projection
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

    #add features to the map
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.COASTLINE)

    # Plot the rectangle representing the defined region
    rectangle = plt.Rectangle((lon_west, lat_south), lon_east - lon_west, lat_north - lat_south,
                            linewidth=2, edgecolor='red', facecolor='none', transform=ccrs.PlateCarree())
    rectangle_A = plt.Rectangle((lon_west_A, lat_south), lon_east_A - lon_west_A, lat_north - lat_south,
                            linewidth=1, alpha=0.5, edgecolor='blue', facecolor='none', transform=ccrs.PlateCarree())
    ax.add_patch(rectangle)
    ax.add_patch(rectangle_A)
    # Set the extent of the map to cover the specified region
    #ax.set_global()
    ax.set_extent([lon_west-40, lon_east+40, lat_south-15, lat_north+15], crs=ccrs.PlateCarree())

    ax.set_title(f'Map of {region_name}')
    plt.show()