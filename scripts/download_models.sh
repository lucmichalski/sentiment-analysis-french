#!/bin/sh

set -x
set -e 

mkdir -p /src/models

wget --no-check-certificate -O /usr/local/bin/gdrivedl 'https://f.mjh.nz/gdrivedl'
chmod +x /usr/local/bin/gdrivedl
gdrivedl https://drive.google.com/open?id=1GL0zdThuAECX6zo1rA_ExKha1CPu1h_h /src/models/camembert_sentiment.tar.xz
tar xf camembert_sentiment.tar.xz
