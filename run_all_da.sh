#!/bin/sh
set -x
timestamp=$(date '+%FT%T')

train_model() {
  ./da_forecast.py \
    --learn=$2\
    --model results/$2/$1.model \
    --prediction results/$2/train-$1.csv \
    --feature datasets/train/feature-$1.csv \
    --target datasets/train/target-$1.csv
}

test_model() {
  ./da_forecast.py \
    --model results/$2/$1.model \
    --feature datasets/test/feature-$1.csv \
    --prediction results/$2/test-$1.csv
}

validate_model() {
  ./da_validate.py \
    --pairs $1-pairs.csv \
    --scoreboard results/$1-scoreboard-$timestamp.csv
}

visualize_prediction() {
  ./da_visualize.py \
    --prediction results/da/test-$1.csv \
    --target datasets/da_test/target-$1.csv \
    --parse-dates \
    --title $zone \
    --save results/da/test-$1.png
}


zones='
nyiso-CAPITL
nyiso-CENTRL
nyiso-DUNWOD
nyiso-GENESE
nyiso-HUD_VL
nyiso-LONGIL
nyiso-MHK_VL
nyiso-MILLWD
nyiso-NORTH
nyiso-NYC
nyiso-WEST
synthetic
'

run_task() {
  for zone in $zones
  do
    train_model $zone $1
    test_model $zone $1
#    visualize_prediction $zone
  done
  validate_model $1
}

run_task qr
run_task et
