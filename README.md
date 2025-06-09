# MoniTor: Exploiting Large Language Models with Instruction for Online Video Anomaly Detection

<div align="center">

**Shengtian Yang**, **Yue Feng**, **Yingshi Liu**, **Jingrou Zhang**, **Jie Qin**

</div>

<div align="center">
  <img src="https://github.com/user-attachments/assets/4fa0a527-8a79-4778-a62f-beab0a5f94b5" alt="MoniTor Method Overview" width="800">
</div>
Video Anomaly Detection (VAD) aims to locate unusual activities or behaviors within videos. Recently, offline VAD has garnered substantial research attention, which has been invigorated by the progress in large language models (LLMs) and vision-language models (VLMs), offering the potential for a more nuanced understanding of anomalies. 
However, online VAD has seldom received attention due to real-time constraints and computational intensity. 
In this paper, we introduce a novel Memory-based online scoring queue scheme for Training-free VAD (MoniTor), to address the inherent complexities in online VAD. 
Specifically, MoniTor applies a streaming input to VLMs, leveraging the capabilities of pre-trained large-scale models. 
To capture temporal dependencies more effectively, we incorporate a novel prediction mechanism inspired by Long Short-Term Memory (LSTM) networks to ensure that the model can effectively model past states and leverage previous predictions to identify anomalous behaviors, thereby better understanding the current frame. 
Moreover, we design a scoring queue and an anomaly prior to dynamically store recent scores and cover all anomalies in the monitoring scenario, providing guidance for LLMs to distinguish between normal and abnormal behaviors over time.
We evaluate MoniTor on two large datasets (i.e., UCF-Crime and XD-Violence) containing various surveillance and real-world scenarios. 
The results demonstrate that MoniTor outperforms state-of-the-art methods and is competitive with weakly supervised methods without training. Code will be available.

## Setup

We recommend the use of a Linux machine with CUDA compatible GPUs. We used 2x NVIDIA 4090 GPUs with 24GB. We provide a Conda environment to configure the required libraries.

```bash
git clone https://github.com/YsTvT/MoniTor.git
cd MoniTor
```

## Conda

The environment can be installed and activated with:

```bash
conda create --name Moni python=3.10
conda activate Moni
pip install -r requirements.txt
```
# Data

Please download the data, including captions, temporal summaries, indexes with their textual embeddings, and scores for the UCF-Crime and XD-Violence datasets, from the links below:

- 📦 **UCF-Crime Dataset (processed)**  
  [Download UCF-Crime features](https://drive.google.com/file/d/1_7juCgOoWjQruyH3S8_FBqajuRaORmnV/view)  
  Includes: pre-extracted I3D features, timestamps, textual descriptions, and anomaly labels.

- 📦 **XD-Violence Dataset (processed)**  
  [Download XD-Violence features](https://drive.google.com/file/d/1yzDP1lVwPlA_BS2N5Byr1PcaazBklfkI/view)  
  Includes: temporal annotations, video embeddings, and textual captions.

After downloading, extract the contents into the following directory structure:

# Pretrained models

We use **Blip2** as our VLM following [lavad](https://github.com/lucazanella/lavad?tab=readme-ov-file).

# LLM Agent

We use **GLM-4-Flash** as our default LLM agent due to its efficiency and compatibility.  
You are free to use other models from the **GLM** family depending on your requirements.

🔗 [Visit the official GLM page](https://github.com/THUDM/ChatGLM) to explore available models and documentation.  
🔑 You can obtain your API key from the GLM service at [https://open.bigmodel.cn](https://open.bigmodel.cn).

