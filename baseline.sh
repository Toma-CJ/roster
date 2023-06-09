#export CUDA_DEVICE_ORDER=PCI_BUS_ID
#export CUDA_VISIBLE_DEVICES=1

while [[ $# -gt 0 ]]; do
  case $1 in
    -d|--dataset)
      CORPUS="$2"
      shift # past argument
      shift # past value
      ;;
    -e|--eval_dataset)
      EVAL_DATASET="$2"
      shift # past argument
      shift # past value
      ;;
    -te|--train_epochs)
      TRAIN_EPOCHS="$2"
      shift # past argument
      shift # past value
      ;;
    -ts|--tag_scheme)
      TAG_SCHEME="$2"
      shift # past argument
      shift # past value
      ;;
    -em|--ensemble_models)
      ENSEMBLE_MODELS="$2"
      shift # past argument
      shift # past value
      ;;
    -ee|--ensemble_epochs)
      ENSEMBLE_EPOCHS="$2"
      shift # past argument
      shift # past value
      ;;
    *)
      echo "Invalid argument: $1"
      shift # past argument
      ;;
  esac
done

SEED=30
TEMP_DIR=tmp_${CORPUS}_$SEED
OUT_DIR=out_$CORPUS
mkdir -p $TEMP_DIR
mkdir -p $OUT_DIR
mkdir -p $OUT_DIR/data/$EVAL_DATASET

eval "$(conda shell.bash hook)"
conda activate 2yp

python -u src/train.py --data_dir data/$CORPUS \
    --output_dir $OUT_DIR --temp_dir $TEMP_DIR \
    --pretrained_model roberta-base --tag_scheme $TAG_SCHEME --max_seq_length 120 \
    --train_batch_size 32 --gradient_accumulation_steps 1 --eval_batch_size 64 \
    --noise_train_lr 3e-5 --ensemble_train_lr 1e-5 --self_train_lr 5e-7 \
    --noise_train_epochs $TRAIN_EPOCHS --ensemble_train_epochs $ENSEMBLE_EPOCHS --self_train_epochs 5 \
    --noise_train_update_interval 60 --self_train_update_interval 100 \
    --dropout 0.1 --warmup_proportion=0.1 --seed $SEED \
    --q 0.7 --tau 0.7 --num_models $ENSEMBLE_MODELS \
    --do_train --do_eval --eval_on "valid" | tee $OUT_DIR/train_log.txt

python -u src/train.py --data_dir data/$EVAL_DATASET \
    --output_dir $OUT_DIR --temp_dir $TEMP_DIR \
    --pretrained_model roberta-base --tag_scheme $TAG_SCHEME --max_seq_length 120 \
    --do_eval --eval_on "test" | tee $OUT_DIR/test_log.txt

rm -rf $TEMP_DIR
    
    
