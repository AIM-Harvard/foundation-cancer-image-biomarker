if [ ! -d "supervised" ]; then
  mkdir supervised
fi

cd "supervised"
if [ ! -d "task2" ]; then
  mkdir task2
fi

wget https://www.dropbox.com/s/zypjv5k6wohxwn3/task2_foundation_finetuned.torch?dl=1 -O supervised/task2_foundation_finetuned.torch
wget https://www.dropbox.com/s/7pu0irhlubcp6ji/task2_supervised_finetuned.torch?dl=1 -O supervised/task2_supervised_finetuned.torch
wget https://www.dropbox.com/s/44f7stxvxu9n2d1/task2_supervised.torch?dl=1 -O supervised/task2_supervised.torch
