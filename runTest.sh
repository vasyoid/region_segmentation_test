#!/bin/bash

for d in res/*/;
do
  echo $d
  test/run "$d"picture.png "$d"mask.png "$d"test.png;
done