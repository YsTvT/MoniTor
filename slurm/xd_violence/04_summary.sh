#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --array=0-0%1
#SBATCH --output=output/04_query_llm_ucf_crime_%A_%a.out
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH=/root/autodl-tmp/lavad-main

# Set the XD Violence directory
xd_violence_dir="/your/path/to/xd_violence"

# Set paths
root_path="${xd_violence_dir}/frames"
annotationfile_path="${xd_violence_dir}/annotations/anomaly_test.txt"
llm_model_name="llama-2-13b-chat"
batch_size=1
frame_interval=16
api_key="your_api_key"
num_job=180

context_prompt="your_context_prompt"
"""
"How would you rate this scene on a scale from 0 to 1, where 0 represents a normal scene and 1 represents a scene involving suspicious or potential criminal activities, such as:
Abuse (inflicting improper harm or misconduct on others)
Car Accident (collision events involving motor vehicles on roads that may cause property damage or personal injury)
Explosion (phenomenon where substances release energy violently due to chemical reactions or physical processes)
Fighting (physical conflicts between two or more parties due to disputes, revenge, or other reasons)
Riot (a group of people protesting or resisting against the government or other organizations in a violent manner)
Shooting (act of firing with firearms, possibly involving murder or assault)"

"If you are law enforcement, how would you score the described scene on a scale from 0 to 1, where 0 represents a standard scene and 1 represents a scene involving suspicious or potential criminal activities, such as abuse, car accidents, explosions, fighting, riots, shooting, etc.?"
"""

format_prompt="your_format_prompt"
#"Please provide your response in the form of a Python list, and only answer with one number from the following provided number list [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], without any textual explanation. It should start with '[' and end with ']'."

index_name="flan-t5-xxl"
echo "Processing index: $index_name"

captions_dir="$xd_violence_dir/captions/clean/$index_name/"
dir_name="your_dir_name"
output_scores_dir="${xd_violence_dir}/scores/raw/${llm_model_name}/${index_name}/${dir_name}/"
output_summary_dir="${xd_violence_dir}/captions/summary/${llm_model_name}/$index_name/"

python -u /root/autodl-tmp/lavad-main/src/models/xd_04.py \
    --root_path "$root_path" \
    --annotationfile_path "$annotationfile_path" \
    --batch_size "$batch_size" \
    --frame_interval "$frame_interval" \
    --context_prompt "$context_prompt" \
    --format_prompt "$format_prompt" \
    --output_scores_dir "$output_scores_dir" \
    --captions_dir "$captions_dir" \
    --api_key "$api_key" \
    --num_jobs "$num_job"
