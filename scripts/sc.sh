MODEL_NAME_OR_PATH=$1
TOKENIZER_NAME_OR_PATH=$2
K=$3
SEED=$4
PROBLEM_NAME=$5
NUMBER_OF_SUBPROBLEMS=$6
STEPS=${7}
PORT_ID=${8}
DATASET_NAME_OR_PATH=${9}
METHOD_PROMPT=${10}

ARGS="\
    --model_name_or_path $MODEL_NAME_OR_PATH\
    --tokenizer_name_or_path $TOKENIZER_NAME_OR_PATH\
    --dataset_name_or_path $DATASET_NAME_OR_PATH\
    --request_batch_size 2\
    --inference_api openai\
    --api_key <api key>\
    --do_sample\
    --max_samples 10\
    --num_return_sequences 8\
    --num_beams 8\
    --max_new_tokens 1000\
    --temperature 0.7\
    --top_p 0.95\
    --repetition_penalty 1.0\
    --output_dir ./reasoning\
    --k $K\
    --seed $SEED\
    --problem_name $PROBLEM_NAME\
    --method_prompt $METHOD_PROMPT\
    --steps $STEPS\
    --verbose\
    --number_of_subproblems $NUMBER_OF_SUBPROBLEMS\
    "
torchrun \
    --rdzv-backend=c10d\
    --rdzv-endpoint=localhost:$PORT_ID\
    main.py \
    $ARGS \