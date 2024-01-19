if [ ! -d "supervised" ]; then
  mkdir supervised
fi

cd "supervised"
if [ ! -d "task3" ]; then
  mkdir task3
fi

wget https://www.dropbox.com/s/u6uulvijpkd9mf6/task3_foundation_finetuned.torch?dl=1 -O supervised/task3_foundation_finetuned.torch
wget https://www.dropbox.com/s/94512wr5lmhwqgc/task3_supervised_finetuned.torch?dl=1 -O supervised/task3_supervised_finetuned.torch
wget https://www.dropbox.com/s/57ga9isdduz4ea6/task3_supervised.torch?dl=1 -O supervised/task3_supervised.torch
