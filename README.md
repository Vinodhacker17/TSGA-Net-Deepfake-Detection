🛡️ SentinAI: Advanced Deepfake Detection System
SentinAI is a state-of-the-art deep learning framework designed to detect high-fidelity AI-generated images (Deepfakes). Unlike traditional detectors that rely solely on visual artifacts, this project introduces TSGA-Net (Tri-Stream Spatial-Graph-Attention Network), a novel architecture that analyzes images in three dimensions simultaneously: visual texture, attentional focus, and structural consistency.

🚀 Key Features
Novel Architecture: Combines ResNet-18 (Spatial), Self-Attention (Focus), and Graph Attention Networks (GAT) (Structural) into a single hybrid model.

Massive Scale: Trained on 100,000+ images sourced from diverse generative models (Stable Diffusion, Midjourney, StyleGAN).

High Accuracy: Achieved ~95% validation accuracy on the test set.

Interactive UI: Fully deployed web interface using Streamlit with real-time "Anomaly Sensitivity" calibration.

🧠 Model Architecture (TSGA-Net)
The system operates on three parallel streams:

Spatial Stream: Extracts high-frequency texture features using a ResNet-18 backbone.

Attention Stream: Utilizes a Self-Attention mechanism to weigh artifact-prone regions (eyes, lips, hair).

Graph Stream: Transforms image patches into a graph structure and uses GATs to verify geometric consistency.

🛠️ Tech Stack
Framework: PyTorch

Graph Library: PyTorch Geometric

Interface: Streamlit

Training Hardware: NVIDIA Tesla T4 GPUs (via Kaggle)

📂 Installation & Usage
Clone the repository:

Bash

git clone https://github.com/YOUR_USERNAME/SentinAI.git
cd SentinAI
Install dependencies:

Bash

pip install -r requirements.txt
Run the application:

Bash

streamlit run app.py
📄 Research
This project is based on original research conducted by Vinod N, Rudra Pratap Singh, and Prince Kumar Singh.
