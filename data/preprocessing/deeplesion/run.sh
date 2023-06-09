BASEDIR=$(dirname "$0")
cd $BASEDIR

# Unzip the Images_png folder
unzip "$1/DeepLesion/Images_png/*.zip" -d $1/DeepLesion/Images_png

# Convert the PNGs to NIFTI
python process_deeplesion.py $1/DeepLesion/Images_png/Images_png $1/DeepLesion/Images_nifti $1/DeepLesion/DL_info.csv

# Generate the annotations from DL_info.csv and NIFTI files
python process_annotations.py $1/DeepLesion/DL_info.csv ./annotations

# Format the annotations into global coordinates
python format_annotations.py $1
