#!/bin/bash

# Set the directory where your .grb files are located
input_dir="/cluster/projects/nn9348k/Jakub/article/data/daily_Z"

# Iterate over all .grb files in the directory
for grb_file in "$input_dir"/*.grb; do
    # Check if there are any .grb files in the directory
    if [ -e "$grb_file" ]; then
        # Extract the filename without extension
        filename_without_extension=$(basename "$grb_file" .grb)
        
        # Create the output NetCDF filename
        nc_file="${input_dir}/${filename_without_extension}.nc"
        
        # Convert .grb to .nc using cdo
        cdo -R -f nc -t ecmwf copy "$grb_file" "$nc_file"
        
        # Print a message indicating the conversion
        echo "Converted $grb_file to $nc_file"
    fi
done
