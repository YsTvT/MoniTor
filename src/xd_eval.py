import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from src.data.video_record import VideoRecord
from src.utils.vis_utils import visualize_video


def temporal_testing_annotations(temporal_annotation_file):
    annotations = {}
    with open(temporal_annotation_file) as annotations_f:
        for line in annotations_f:
            parts = line.strip().split()
            video_name = str(Path(parts[0]).stem)
            annotation_values = parts[2:]
            annotations[video_name] = annotation_values
    return annotations


def replace_negative_scores_with_previous(scores_dict):
    prev_score = 0.0  # Initialize to 0.0 to replace the first frame -1 case
    for frame_idx in sorted(scores_dict.keys(), key=lambda x: int(x)):
        if scores_dict[frame_idx] == -1:
            scores_dict[frame_idx] = prev_score
        else:
            prev_score = scores_dict[frame_idx]
    return scores_dict


def get_video_labels(video_record, annotations, normal_label):
    video_name = Path(video_record.path).name
    labels = []
    video_annotations = [x for x in annotations[video_name] if x != "-1"]

    # Separate start and stop indices
    start_indices = video_annotations[::2]
    stop_indices = video_annotations[1::2]

    for frame_index in range(video_record.num_frames):
        frame_label = normal_label

        if len(video_record.label) == 1:
            for start_idx, end_idx, label in zip(
                start_indices, stop_indices, video_record.label * len(start_indices)
            ):
                if int(start_idx) <= frame_index + video_record.start_frame <= int(end_idx):
                    frame_label = label
        else:
            video_labels = video_record.label

            # Pad video_labels if it's shorter than start_indices
            if len(video_labels) < len(start_indices):
                last_label = [video_record.label[-1]] * (len(start_indices) - len(video_labels))
                video_labels.extend(last_label)

            for start_idx, end_idx, label in zip(start_indices, stop_indices, video_labels):
                if int(start_idx) <= frame_index + video_record.start_frame <= int(end_idx):
                    frame_label = label

        labels.append(frame_label)

    return labels


def calculate_weighted_scores(scores_dict, num_neighbors, frame_interval):
    scores = []

    for frame_idx in scores_dict.keys():
        frame_idx = int(frame_idx)
        frame_scores = []

        for i in range(max(0, frame_idx - num_neighbors + 1), frame_idx + 1):
            if str(i) in scores_dict:
                # Ensure only numerical values are added to frame_scores
                if isinstance(scores_dict[str(i)], dict) and "new_score" in scores_dict[str(i)]:
                    frame_score = scores_dict[str(i)]["new_score"]
                    if isinstance(frame_score, (int, float)):  # Ensure frame_score is numerical
                        frame_scores.append(frame_score)
                elif isinstance(scores_dict[str(i)], (int, float)):
                    frame_scores.append(scores_dict[str(i)])

        avg_score = np.mean(frame_scores) if frame_scores else 0.0
        scores.append(avg_score)

    scores = np.repeat(scores, frame_interval)
    return scores


def save_metric(output_dir, metric_name, num_neighbors, metric_value):
    with open(output_dir / f"{metric_name}_nn_{num_neighbors}.txt", "w") as f:
        f.write(f"{metric_value}\n")


def load_json_file(file_path):
    """Helper function to load a JSON file and handle errors."""
    try:
        with open(file_path) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON for {file_path}: {e}")
        return None  # Return None to indicate failure


def main(
    root_path,
    annotationfile_path,
    temporal_annotation_file,
    raw_scores_dir,
    captions_dir,
    output_dir,
    frame_interval,
    normal_label,
    num_neighbors,
    without_labels,
    visualize,
    video_fps,
):
    raw_scores_dir = Path(raw_scores_dir)
    captions_dir = Path(captions_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the temporal annotations
    if not without_labels:
        annotations = temporal_testing_annotations(temporal_annotation_file)

    # Load video records from the annotation file
    video_list = [VideoRecord(x.strip().split(), root_path) for x in open(annotationfile_path)]

    flat_scores = []
    flat_labels = []

    for video in video_list:
        video_name = Path(video.path).name

        # Load the scores and captions
        video_scores_path = raw_scores_dir / f"{video_name}.json"
        video_captions_path = captions_dir / f"{video_name}.json"

        video_scores_dict = load_json_file(video_scores_path)
        if video_scores_dict is None:
            continue  # Skip this video if there is an error

        video_scores_dict = replace_negative_scores_with_previous(video_scores_dict)

        video_captions = load_json_file(video_captions_path)
        if video_captions is None:
            continue  # Skip this video if there is an error

        # Get video labels
        if without_labels:
            video_labels = []
        else:
            video_labels = get_video_labels(video, annotations, normal_label)

        video_scores = calculate_weighted_scores(
            video_scores_dict, num_neighbors, frame_interval
        )

        video_scores = video_scores[: video.num_frames]

        flat_scores.extend(video_scores)
        if not without_labels:
            flat_labels.extend(video_labels)

        if visualize:
            visualize_video(
                video_name,
                [],
                video_scores,
                video_captions,
                video.path,
                video_fps,
                output_dir / f"{video_name}.mp4",
                normal_label,
                "{:06d}.jpg",
                None,
            )

    flat_scores = np.array(flat_scores)

    if not without_labels:
        flat_labels = np.array(flat_labels)

        # Ensure lengths match
        if len(flat_labels) != len(flat_scores):
            print(f"Warning: Length mismatch between labels ({len(flat_labels)}) and scores ({len(flat_scores)})")
            min_length = min(len(flat_labels), len(flat_scores))
            flat_labels = flat_labels[:min_length]
            flat_scores = flat_scores[:min_length]

        flat_binary_labels = flat_labels != normal_label

        # Compute ROC AUC score
        fpr, tpr, threshold = roc_curve(flat_binary_labels, flat_scores)
        roc_auc = auc(fpr, tpr)
        print(f"ROC AUC: {roc_auc}")
        save_metric(output_dir, "roc_auc", num_neighbors, roc_auc)

        # Compute precision-recall curve
        precision, recall, th = precision_recall_curve(flat_binary_labels, flat_scores)
        pr_auc = auc(recall, precision)
        print(f"PR AUC: {pr_auc}")
        save_metric(output_dir, "pr_auc", num_neighbors, pr_auc)


def parse_args():
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--annotationfile_path", type=str, required=True)
    parser.add_argument("--temporal_annotation_file", type=str)
    parser.add_argument("--raw_scores_dir", type=str, required=True)
    parser.add_argument("--captions_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    # Optional arguments with defaults
    parser.add_argument("--frame_interval", type=int, default=16)
    parser.add_argument("--normal_label", type=int)
    parser.add_argument("--num_neighbors", type=int, default=10)

    parser.add_argument("--without_labels", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--video_fps", type=float)

    args = parser.parse_args()
    if args.temporal_annotation_file is None and not args.without_labels:
        parser.error("--temporal_annotation_file is required when --without_labels is not used")
    if args.visualize and args.video_fps is None:
        parser.error("--video_fps is required when --visualize is used")

    return args


if __name__ == "__main__":
    args = parse_args()
    main(
        args.root_path,
        args.annotationfile_path,
        args.temporal_annotation_file,
        args.raw_scores_dir,
        args.captions_dir,
        args.output_dir,
        args.frame_interval,
        args.normal_label,
        args.num_neighbors,
        args.without_labels,
        args.visualize,
        args.video_fps,
    )
