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
echo "Evaluation dataset:${CORPUS}:"

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
    
