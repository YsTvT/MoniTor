# MoniTor: Exploiting Large Language Models with Instruction for Online Video Anomaly Detection
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

##Setup

We recommend the use of a Linux machine with CUDA compatible GPUs. We used 2x NVIDIA A100 GPUs with 64GB. We provide both a Conda environment and a Dockerfile to configure the required libraries.

```bash
[git clone https://github.com/lucazanella/lavad.git](https://github.com/YsTvT/MoniTor.git)
cd MoniTor
```

## Conda

The environment can be installed and activated with:

```bash
conda create --name Moni python=3.10
conda activate Moni
pip install -r requirements.txt
```
