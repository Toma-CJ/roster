#export CUDA_DEVICE_ORDER=PCI_BUS_ID
#export CUDA_VISIBLE_DEVICES=1

while [[ $# -gt 0 ]]; do
  case $1 in
    -d|--dataset)
      CORPUS="$2"
      shift # past argument
      shift # past value
      ;;
    *)
      echo "Invalid argument: $1"
      shift # past argument
      ;;
  esac
done
echo "dataset:${CORPUS}:"

SEED=30
TEMP_DIR=tmp_${CORPUS}_$SEED
OUT_DIR=out_$CORPUS
mkdir -p $TEMP_DIR
mkdir -p $OUT_DIR

eval "$(conda shell.bash hook)"
conda activate 2yp

python -u src/train.py --data_dir data/final/$CORPUS \
    --output_dir $OUT_DIR --temp_dir $TEMP_DIR \
    --pretrained_model roberta-base --tag_scheme 'iob' --max_seq_length 120 \
    --do_eval --eval_on "test" | tee $OUT_DIR/test_log.txt

    #    --train_batch_size 32 --gradient_accumulation_steps 2 --eval_batch_size 64 \
    # --noise_train_lr 3e-5 --ensemble_train_lr 1e-5 --self_train_lr 5e-7 \
    # --noise_train_epochs 5 --ensemble_train_epochs 10 --self_train_epochs 5 \
    # --noise_train_update_interval 60 --self_train_update_interval 100 \
    # --dropout 0.1 --warmup_proportion=0.1 --seed $SEED \
    # --q 0.7 --tau 0.7 --num_models 5 \
    
