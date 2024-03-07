# Read the csv file
data=$(cat nsclc_radiogenomics.csv) 

# Get series directories
series_dirs=$(echo "$data" | awk -F',' '{print $5}') 

# Check if directory exists, if not create it
if [ ! -d "$1" ]; then
  mkdir -p $1
fi


# Iterate over column data and create folder
for item in $series_dirs
do
    if [ ! -d "$1/$item" ]; then
        mkdir "$1/$item"
    fi
done

download_commands=$(echo "$data" | awk -F',' 'NR>1{print $NF}') # Get the last column, skip the first row
cd $1
IFS=$'\n' # Set Internal Field Separator to newline
for command in $download_commands
do
    echo $command >> radio_manifest.s5cmd
done

s5cmd --no-sign-request --endpoint-url https://storage.googleapis.com run radio_manifest.s5cmd