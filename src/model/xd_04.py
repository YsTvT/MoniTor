# #OURS
import argparse
import json
import re
import time
from pathlib import Path
from typing import List
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.data.video_record import VideoRecord
from zhipuai import ZhipuAI

class GLMAnomalyScorer:
    def __init__(
        self,
        api_key,
        root_path,
        batch_size,
        frame_interval,
        context_prompt,
        format_prompt,
        output_scores_dir,
        captions_dir,
    ):
        self.api_key = api_key
        self.root_path = root_path
        self.batch_size = batch_size
        self.frame_interval = frame_interval
        self.context_prompt = context_prompt
        self.format_prompt = format_prompt
        self.output_scores_dir = output_scores_dir
        self.captions_dir = captions_dir
        self.client = ZhipuAI(api_key=self.api_key)

        # Initialize the score queue
        self.score_queue = {score: [] for score in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
        # Initialize prediction memory
        self.prediction1 = "No prediction available"  # Used to store the prediction of the current frame
        self.prediction2 = "No prediction available"  # Used to store the prediction of the next frame
        # Initialize short-term and long-term memory
        self.long_term_memory = []  # Long-term memory, stores descriptions of video frames from the previous 5 seconds
        self.short_term_memory = []  # Short-term memory, stores descriptions of the previous 2 frames

    def _parse_score(self, response):
        pattern = r"\[(\d+(?:\.\d+)?)\]"
        match = re.search(pattern, response)
        if match:
            score = float(match.group(1))
        else:
            alternative_pattern = r"score is (\d+(?:\.\d+)?)"
            match = re.search(alternative_pattern, response)
            score = float(match.group(1)) if match else -1

        if score != -1:
            allowed_scores = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
            score = min(allowed_scores, key=lambda x: abs(x - score))
        return score

    def _update_score_queue(self, frame_idx, score):
        # Update the queue with the latest frame index for the given score
        if score in self.score_queue:
            self.score_queue[score].append(frame_idx)

    def _lstm_summarize(self, texts):
        # Simulate LSTM-based summarization by combining texts into a summary
        combined_text = " ".join(texts)
        summary = self.client.chat.completions.create(
            model="glm-4-flash",
            messages=[
                {"role": "system", "content": "Please summarize the following descriptions into a concise sentence."},
                {"role": "user", "content": combined_text}
            ],
            stream=False,
            timeout=30,
            temperature=0.6
        ).choices[0].message.content.strip()
        return summary

    def _calculate_similarity(self, text1, text2):
        # Use TF-IDF Vectorizer to convert text to vectors
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([text1, text2])

        # Calculate cosine similarity between the vectors
        similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        similarity_score = similarity_matrix[0][0]

        return similarity_score

    def _prepare_memory_summaries(self, captions, current_idx):
        # Prepare long-term memory (summarizing previous 5 seconds of descriptions)
        long_term_captions = [
            captions[str(i)] for i in range(max(0, current_idx - 5 * self.frame_interval), current_idx, self.frame_interval)
            if str(i) in captions
        ]
        # Apply a forgetting mechanism based on similarity
        filtered_long_term_captions = []
        for caption in long_term_captions:
            if not filtered_long_term_captions:
                filtered_long_term_captions.append(caption)
            else:
                similarity = self._calculate_similarity(filtered_long_term_captions[-1], caption)
                forgetting_threshold = 0.5  # Set threshold for forgetting based on similarity
                if similarity < forgetting_threshold:
                    filtered_long_term_captions.append(caption)

        self.long_term_memory = filtered_long_term_captions[-5:]  # Limit long-term memory to 5 descriptions
        long_term_summary = self._lstm_summarize(self.long_term_memory) if self.long_term_memory else "No long-term memory"

        # Prepare short-term memory (summarizing the last 2 frames)
        short_term_captions = [
            captions[str(current_idx - i * self.frame_interval)] for i in range(1, 3)
            if str(current_idx - i * self.frame_interval) in captions
        ]
        self.short_term_memory = short_term_captions[-2:]  # Keep only the latest 2 frame descriptions
        short_term_summary = self._lstm_summarize(self.short_term_memory) if self.short_term_memory else "No short-term memory"

        return long_term_summary, short_term_summary

    def _prepare_score_prompts(self, captions):
        # Prepare score queue information to be used as a prompt for scoring
        score_prompts = []
        for score, frames in self.score_queue.items():
            if frames:
                frame_idx = frames[-1]
                caption = captions[str(frame_idx)]
                score_prompts.append(f"Event with score {score}: '{caption}'.")
        return "\n".join(score_prompts)

    def _prepare_dialogs(self, captions, batch_frame_idxs):
        # Get score queue information as part of the prompt
        score_prompts = self._prepare_score_prompts(captions)
        prompt = self.context_prompt + " " + self.format_prompt + "\n" + score_prompts

        scoring_dialogs = []  # Store scoring dialogs

        for idx in batch_frame_idxs:
            # Current frame caption
            current_caption = f"This is the description of the current frame: '{captions[str(idx)]}'."

            # Prepare long-term and short-term memory summaries
            long_term_summary, short_term_summary = self._prepare_memory_summaries(captions, idx)
            memory_info = f"Long-term memory contains a summary of events from the previous five seconds: {long_term_summary}. Short-term memory contains descriptions of the previous two frames: {short_term_summary}."

            # **Construct prediction dialog**
            if idx - self.frame_interval >= 0:  # Ensure there is a previous frame
                previous_caption = captions.get(str(idx - self.frame_interval), "No previous caption available")
                prediction_prompt = ("If you are law enforcement, please predict what might happen next in the described scene, considering potential suspicious activities or behaviors, "
                                    "such as abuse, arrest, arson, assault, burglary, disturbing the peace, explosion, fighting, robbery, shooting, stealing, shoplifting, or vandalism. "
                                    "Please provide a concise prediction based on the current context.\n"
                                    "Please provide the prediction in a concise sentence, focusing on describing the behavior or event that might occur next in the scene, avoiding any additional explanations.")
                prediction_response = self.client.chat.completions.create(
                    model="glm-4-flash",
                    messages=[
                        {"role": "system", "content": prediction_prompt},
                        {"role": "user", "content": f"The description of the previous frame is: '{previous_caption}', please provide the description of the next frame."}
                    ],
                    stream=False,
                    timeout=60,
                    temperature=0.6
                )
                prediction_result = prediction_response.choices[0].message.content.strip()
                #print(f"Prediction result: {prediction_result}")
            else:
                prediction_result = "No previous frame available for prediction"

            # Store prediction for use in the next frame
            self.prediction2 = prediction_result

            # **Construct scoring dialog including prediction result**
            scoring_prompt = (prompt + f"\n{memory_info}\nPrediction from the previous frame: '{self.prediction1}'")

            scoring_dialog = [
                {"role": "system", "content": scoring_prompt},
                {"role": "user", "content": f"{current_caption} "}
            ]
            scoring_dialogs.append(scoring_dialog)

            # Update predictions: prediction1 becomes prediction2 for the next frame
            self.prediction1 = self.prediction2

        return scoring_dialogs

    def _score_temporal_summaries(self, video, temporal_captions):
        video_scores = {}
        previous_scores = {}
        fy = {}
        frame_idxs = list(range(0, video.num_frames, self.frame_interval))

        for batch_start in tqdm(range(0, len(frame_idxs), self.batch_size), desc=f"Processing {video.path}", unit="batch"):
            batch_end = min(batch_start + self.batch_size, len(frame_idxs))
            batch_frame_idxs = frame_idxs[batch_start:batch_end]
            scoring_dialogs = self._prepare_dialogs(temporal_captions, batch_frame_idxs)

            results = []
            for dialog, frame_idx in zip(scoring_dialogs, batch_frame_idxs):
                try:
                    response = self.client.chat.completions.create(
                        model="glm-4-flash",
                        messages=dialog,
                        stream=False,
                        timeout=60,  # Increase timeout
                        temperature=0.6  # Lower temperature to reduce randomness
                    )
                except Exception as e:
                    print(f"API request failed during scoring: {e}")
                    continue

                result_content = response.choices[0].message.content
                results.append({"generation": {"content": result_content, "prompt": dialog[0]["content"], "caption": dialog[1]["content"]}})

            # Process scoring results
            for result, frame_idx in zip(results, batch_frame_idxs):
                response = result["generation"]["content"]
                current_score = self._parse_score(response)
                previous_score = video_scores.get(str(frame_idx - self.frame_interval), None)
                if previous_score is not None and previous_score != -1:
                    score = round(0.7 * current_score + 0.3 * previous_score, 1)
                else:
                    score = round(current_score, 1)

                self._update_score_queue(frame_idx, score)  # Update score queue

                fy[str(frame_idx)] = {'new_score': score, 'raw_score': response, 'caption': result["generation"]["caption"], 'prompt': result["generation"]["prompt"]}
                video_scores[str(frame_idx)] = score
                previous_scores[str(frame_idx)] = score

        # Smoothing scores
        sorted_indices = sorted(video_scores.keys(), key=lambda x: int(x))
        scores = [video_scores[idx] for idx in sorted_indices]
        smoothed_scores = np.convolve(scores, np.ones(3) / 3, mode='valid')
        smoothed_scores = [round(score, 1) for score in smoothed_scores]
        for i, idx in enumerate(sorted_indices[:len(smoothed_scores)]):
            video_scores[idx] = smoothed_scores[i]

        # Update smoothed scores
        for i, idx in enumerate(sorted_indices[:len(smoothed_scores)]):
            fy[idx]['smooth_score'] = smoothed_scores[i]

        return fy

    def process_video(self, video):
        video_name = Path(video.path).name
        temporal_captions_path = Path(self.captions_dir) / f"{video_name}.json"
        try:
            with open(temporal_captions_path) as f:
                temporal_captions = json.load(f)
        except FileNotFoundError:
            print(f"Captions not found for video {video_name}")
            return
        except json.JSONDecodeError:
            print(f"Failed to parse captions for video {video_name}")
            return

        video_scores = self._score_temporal_summaries(video, temporal_captions)
        output_path = Path(self.output_scores_dir) / f"{video_name}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(video_scores, f, indent=4)

        print(f"Scores successfully saved to {output_path}")

def process_video_parallel(video, scorer_params):
    scorer = GLMAnomalyScorer(**scorer_params)
    scorer.process_video(video)

def run(
    root_path,
    annotationfile_path,
    batch_size,
    frame_interval,
    context_prompt,
    format_prompt,
    output_scores_dir,
    captions_dir,
    api_key,
    num_jobs,
):
    video_list = [VideoRecord(x.strip().split(), root_path) for x in open(annotationfile_path)]

    scorer_params = {
        "api_key": api_key,
        "root_path": root_path,
        "batch_size": batch_size,
        "frame_interval": frame_interval,
        "context_prompt": context_prompt,
        "format_prompt": format_prompt,
        "output_scores_dir": output_scores_dir,
        "captions_dir": captions_dir,
    }

    with ProcessPoolExecutor(max_workers=num_jobs) as executor:
        executor.map(process_video_parallel, video_list, [scorer_params] * len(video_list))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--annotationfile_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--frame_interval", type=int, default=16)
    parser.add_argument("--context_prompt", type=str)
    parser.add_argument("--format_prompt", type=str)
    parser.add_argument("--output_scores_dir", type=str)
    parser.add_argument("--captions_dir", type=str)
    parser.add_argument("--api_key", type=str)
    parser.add_argument("--num_jobs", type=int, default=1)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(
        root_path=args.root_path,
        annotationfile_path=args.annotationfile_path,
        batch_size=args.batch_size,
        frame_interval=args.frame_interval,
        context_prompt=args.context_prompt,
        format_prompt=args.format_prompt,
        output_scores_dir=args.output_scores_dir,
        captions_dir=args.captions_dir,
        api_key=args.api_key,
        num_jobs=args.num_jobs,
    )
