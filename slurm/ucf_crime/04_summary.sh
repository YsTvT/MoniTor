#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --array=0-0%1
#SBATCH --output=output/04_query_llm_ucf_crime_%A_%a.out

#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
#export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

export CUDA_VISIBLE_DEVICES=0,1

# Set the UCF Crime directory
ucf_crime_dir="/your/path/to/ucf_crime"

# Set paths
root_path="${ucf_crime_dir}/frame"
annotationfile_path="${ucf_crime_dir}/annotations/test.txt"
llm_model_name="llama-2-13b-chat"
batch_size=1
frame_interval=16
api_key="your_api_key"
num_job=200
context_prompt= "your_context_prompt(English/Chinese)"
#"If you were a law enforcement agency, how would you rate the described scenes on a scale from 0 to 1, where 0 represents a standard scene and 1 represents a scene involving suspicious activities such as abuse (intentionally harming or mistreating others), arrest (legally detaining someone), arson (deliberately setting fire), assault (physical attack on someone), burglary (illegally entering with the intent to commit a crime), disorderly conduct (disruptive or destructive behavior), explosion (violent release of energy), fighting (violent confrontation), robbery (unlawfully taking property), shooting (firing a gun), shoplifting (stealing from a retail store), theft (taking someone else’s property without permission), or vandalism (deliberate destruction of property)?"(Recommended)

format_prompt="your_format_prompt"
#(Please provide the response in the form of a Python list and respond with only one number in the provided list below [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] without any textual explanation. It should begin with '[' and end with ']'.)(Recommended)
prediction_prompt="your_prediction_prompt(English/Chinese)"
predict_format_prompt="your_predict_format_prompt"
summary_prompt="your_summary_prompt"

index_name="opt-6.7b-coco+opt-6.7b+flan-t5-xxl+flan-t5-xl+flan-t5-xl-coco"
echo "Processing index: $index_name"
captions_dir="${ucf_crime_dir}/captions/summary/${llm_model_name}/${index_name}/"
dir_name="prior+prediction"


# Activate the virtual environment
VENV_DIR="/path/to/venv/lavad"
# shellcheck source=/dev/null
#source "$VENV_DIR/bin/activate"

output_scores_dir="${ucf_crime_dir}/scores/raw/${llm_model_name}/${index_name}/${dir_name}/"

output_summary_dir="${ucf_crime_dir}/captions/summary/${llm_model_name}/$index_name/"

# Run the Python script with the specified parameters
torchrun \
    --nproc_per_node 2 --nnodes 1 -m src.models.04new \
    --root_path "$root_path" \
    --annotationfile_path "$annotationfile_path" \
    --batch_size "$batch_size" \
    --frame_interval "$frame_interval" \
    --context_prompt "$context_prompt" \
    --format_prompt "$format_prompt" \
    --output_scores_dir "$output_scores_dir" \
    --captions_dir "$captions_dir"  \
    --api_key "$api_key" \
    --num_job "$num_job"
