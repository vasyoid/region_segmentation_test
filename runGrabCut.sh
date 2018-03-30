#!/bin/bash

for d in res/*/;
do
  grabcut/grabcut "$d"picture.png "$d"grabcut.png;
done