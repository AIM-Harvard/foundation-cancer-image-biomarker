# This script downloads raw data for DeepLesion from AcademicTorrents
conda install -c conda-forge aria2
aria2c https://academictorrents.com/download/de50f4d4aa3d028944647a56199c07f5fa6030ff.torrent  --seed-time=0  -d $1 # Directory to download is specified with $1, replace with your directory