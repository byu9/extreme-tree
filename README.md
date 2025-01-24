
# Train models on train set

```sh
./da_forecast.py \
  --learn=et \
  --model results/et/nyiso-NYC.model \
  --prediction results/et/train-nyiso-NYC.csv \
  --feature datasets/train/feature-nyiso-NYC.csv \
  --target datasets/train/target-nyiso-NYC.csv
```

```sh
./da_forecast.py \
  --learn=qr \
  --model results/qr/nyiso-NYC.model \
  --prediction results/qr/train-nyiso-NYC.csv \
  --feature datasets/train/feature-nyiso-NYC.csv \
  --target datasets/train/target-nyiso-NYC.csv
```

# Create predictions on test set

```sh
./da_forecast.py \
  --model results/et/nyiso-NYC.model \
  --prediction results/et/test-nyiso-NYC.csv \
  --feature datasets/test/feature-nyiso-NYC.csv
```

```sh
./da_forecast.py \
  --model results/qr/nyiso-NYC.model \
  --prediction results/qr/test-nyiso-NYC.csv \
  --feature datasets/test/feature-nyiso-NYC.csv
```

# Validate test predictions against target

```sh
  ./da_validate.py \
    --pairs et-pairs.csv \
    --scoreboard results/et-scoreboard.csv
```

```sh
  ./da_validate.py \
    --pairs qr-pairs.csv \
    --scoreboard results/qr-scoreboard.csv
```