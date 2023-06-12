BASEDIR=$(dirname "$0")
cd $BASEDIR

# Unzip the Images_png folder
unzip "$1/LUNA16/*.zip" -d $1/LUNA16

python annotations_generator.py ./annotations

python process_luna16.py ./annotations/luna16_annotations.csv $1/LUNA16 --output_dir ./annotations