#!/bin/bash

FILE=hico_20160224_det.tar.gz
EXTR=hico_20160224_det
ID=1dUByzVzM6z1Oq4gENa1-t0FLhr0UtDaS

if [ -d hicodet/$EXTR ]; then
  echo "$EXTR already exists."
  exit 0
fi

echo "Connecting..."

gdown https://drive.google.com/uc?id=$ID

echo "Extracting..."

tar zxf $FILE -C hicodet
rm $FILE

echo "Done."
