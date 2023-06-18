#!/bin/bash

while [[ $# -gt 0 ]]; do
  case $1 in
    -d|--dataset)
      DATASET="$2"
      shift # past argument
      shift # past value
      ;;
    -e|--eval_dataset)
      EVAL_DATASET="$2"
      shift # past argument
      shift # past value
      ;;
    *)
      echo "Invalid argument: $1"
      shift # past argument
      ;;
  esac
done

# Define the number of random combinations
num_combinations=10

# Function to generate a random value based on the specified distribution
function generate_random_value() {
  local distribution=$1
  local min=$2
  local max=$3

  case $distribution in
    "int_uniform")
      echo $((RANDOM % ($max - $min + 1) + $min))
      ;;
    "log_uniform")
      echo "$(awk -v min=$min -v max=$max 'BEGIN{srand(); printf "%.10f", 10^(min+rand()*(log(max/min)/log(10)))}')"
      ;;
    "uniform")
      echo "$(awk -v min=$min -v max=$max 'BEGIN{srand(); printf "%.10f", min+rand()*(max-min)}')"
      ;;
    *)
      echo "Invalid distribution: $distribution"
      exit 1
      ;;
  esac
}

# Generate random combinations
for ((i=1; i<=$num_combinations; i++)); do
  noise_train_epochs=$(generate_random_value 'int_uniform' 3 100)
  ensemble_train_epochs=$(generate_random_value 'int_uniform' 3 100)
  self_train_epochs=$(generate_random_value 'int_uniform' 3 100)

  noise_train_lr=$(generate_random_value 'log_uniform' 1e-7 1e-2)
  ensemble_train_lr=$(generate_random_value 'log_uniform' 1e-7 1e-2)
  self_train_lr=$(generate_random_value 'log_uniform' 1e-7 1e-2)

  q=$(generate_random_value 'uniform' 0 1)
  tau=$(generate_random_value 'uniform' 0 1)

  sbatch main.job -d $DATASET -e $EVAL_DATASET -ts io -te $noise_train_epochs -ee $ensemble_train_epochs \
        -se $self_train_epochs -q $q -t $tau \
        -tlr $noise_train_lr -elr $ensemble_train_lr -slr $self_train_lr
done
