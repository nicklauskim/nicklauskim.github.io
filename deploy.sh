#!/bin/bash

quarto render
cp -r _site/* .
rm -rf _site
git add .
git commit -m "Deploy updated site"
git push origin main

