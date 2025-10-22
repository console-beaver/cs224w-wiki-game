#!/bin/bash

mkdir dataset
cd dataset
echo downloading: Directed English Wikipedia hyperlink network...
wget https://snap.stanford.edu/data/enwiki-2013.txt.gz
echo download finished
echo unzipping download...
gunzip enwiki-2013.txt.gz
echo unzip finished
echo downloading: Names of web pages...
wget https://snap.stanford.edu/data/enwiki-2013-names.csv.gz
echo download finished
echo unzipping download...
gunzip enwiki-2013-names.csv.gz
echo unzip finished
cd ..
