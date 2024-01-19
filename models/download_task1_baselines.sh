if [ ! -d "supervised" ]; then
  mkdir supervised
fi

cd "supervised"
if [ ! -d "task1" ]; then
  mkdir task1
fi

wget https://www.dropbox.com/s/a7sjdigfa29355y/task1_foundation_finetuned.torch?dl=1 -O supervised/task1_foundation_finetuned.torch
wget https://www.dropbox.com/s/bo0scqq1g80li22/task1_supervised.torch?dl=1 -O supervised/task1_supervised.torch
