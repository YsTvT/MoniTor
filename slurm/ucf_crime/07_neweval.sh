#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --array=0-0%1
#SBATCH --output=output/07_eval_ucf_crime_%A_%a.out

# Set the UCF Crime directory
ucf_crime_dir="/your/path/to/ucf_crime"

# Set paths
root_path="${ucf_crime_dir}/frame"
annotationfile_path="${ucf_crime_dir}/annotations/test.txt"
temporal_annotation_file="${ucf_crime_dir}/annotations/Temporal_Anomaly_Annotation_for_Testing_Videos.txt"
llm_model_name="llama-2-13b-chat"
index_name="opt-6.7b-coco+opt-6.7b+flan-t5-xxl+flan-t5-xl+flan-t5-xl-coco"
frame_interval=16
num_neighbors=10
normal_label=7
video_fps=30

context_prompt="your_context_prompt"
#"If you were a law enforcement agency, how would you rate the scene described on a scale from 0 to 1, with 0 representing a standard scene and 1 denoting a scene with suspicious activities?"(Recommended)

exp_id="2035605_002"

dir_name="your_dir_name"

# Activate the virtual environment
VENV_DIR="/path/to/venv/lavad"
# shellcheck source=/dev/null
#source "$VENV_DIR/bin/activate"
# name1="Salesforce"
# model_name="blip2-opt-6.7b-coco"

# captions_dir="$ucf_crime_dir/captions/raw/${name1}/${model_name}/"
captions_dir="${ucf_crime_dir}/captions/clean/${index_name}/"
# Evaluate the AUC-ROC of clip-level scores assigned by the LLM after anomaly score refinement
# captions_dir="$ucf_crime_dir/captions/raw/${name1}/${model_name}/"
raw_scores_dir="${ucf_crime_dir}/scores/raw/${llm_model_name}/${index_name}/${dir_name}/"
echo "Processing scores: $raw_scores_dir"
output_dir="${ucf_crime_dir}/scores/raw/${llm_model_name}/${index_name}/${dir_name}/"

python -m src.neweval \
    --root_path "$root_path" \
    --annotationfile_path "$annotationfile_path" \
    --temporal_annotation_file "$temporal_annotation_file" \
    --raw_scores_dir "$raw_scores_dir" \
    --captions_dir "$captions_dir" \
    --output_dir "$output_dir" \
    --frame_interval "$frame_interval" \
    --normal_label "$normal_label" \
    --num_neighbors "$num_neighbors" \
    --video_fps "$video_fps"
