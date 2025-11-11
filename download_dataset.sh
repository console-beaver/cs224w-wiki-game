#!/bin/bash

mkdir -p dataset
cd dataset
echo downloading: Directed English Wikipedia hyperlink network...
curl -O https://snap.stanford.edu/data/enwiki-2013.txt.gz
echo download finished
echo unzipping download...
gunzip enwiki-2013.txt.gz
echo unzip finished
echo downloading: Names of web pages...
curl -O https://snap.stanford.edu/data/enwiki-2013-names.csv.gz
echo download finished
echo unzipping download...
gunzip enwiki-2013-names.csv.gz
echo unzip finished
cd ..
mkdir -p pickles
mkdir -p embeddings
