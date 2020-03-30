#!/bin/sh
set 
echo "Downloading..."

if [ ! -f "ama_32k_tokenizer.model" ]; then
   wget --no-check-certificate https://storage.googleapis.com/keymoji-data/emotions-models/ama_32k_tokenizer.model
else
  echo "already downloaded ama_32k_tokenizer.model"
fi

if [ ! -f "optimized_model.pth" ]; then
  wget --no-check-certificate https://storage.googleapis.com/keymoji-data/emotions-models/optimized_model.pth
else
  echo "already downloaded optimized_model.pth"
fi

echo "Done!"
