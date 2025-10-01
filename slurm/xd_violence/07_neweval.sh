#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --array=0-0%1
#SBATCH --output=output/07_eval_ucf_crime_%A_%a.out

# Set the UCF Crime directory
xd_violence_dir="/your/path/to/xd_violence"

# Set paths
root_path="${xd_violence_dir}/frames"
annotationfile_path="${xd_violence_dir}/annotations/anomaly_test.txt"
temporal_annotation_file="${xd_violence_dir}/annotations/temporal_anomaly_annotation_for_testing_videos.txt"
llm_model_name="llama-2-13b-chat"
index_name="flan-t5-xxl"
frame_interval=16
num_neighbors=10
normal_label=4
video_fps=24

context_prompt="your_context_prompt"
#"How would you rate the scene described on a scale from 0 to 1, with 0 representing a standard scene and 1 denoting a scene with suspicious or potentially criminal activities?"(Recommended)

exp_id="2039739_001"

dir_name="your_dir_name"

# Activate the virtual environment
VENV_DIR="/path/to/venv/lavad"
# shellcheck source=/dev/null
#source "$VENV_DIR/bin/activate"
captions_dir="${xd_violence_dir}/captions/clean_summary/${llm_model_name}/$index_name/"
raw_scores_dir="${xd_violence_dir}/scores/raw/${llm_model_name}/${index_name}/${dir_name}/"
output_dir="${xd_violence_dir}/scores/refined/${llm_model_name}/${index_name}/${dir_name}/"

echo "Processing scores: $raw_scores_dir"
output_dir="${ucf_crime_dir}/scores/raw/${llm_model_name}/${index_name}/${dir_name}/"

python -m src.xd_eval \
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
