# This script downloads raw data for DeepLesion from AcademicTorrents
conda install -c conda-forge aria2
aria2c https://academictorrents.com/download/58b053204337ca75f7c2e699082baeb57aa08578.torrent  --seed-time=0 -d $1 # Directory to download is specified with $1, replace with your directory