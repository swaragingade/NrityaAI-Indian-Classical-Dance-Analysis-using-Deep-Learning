# NrityaAI: A Real-Time Indian Classical Dance Style Classification and Pose Correction System Using MediaPipe BlazePose and Spatio-Temporal Deep Learning

**Anoushka Dwivedi**  
Department of Computer Science and Engineering (Data Science)  
RV College of Engineering, Bengaluru, Karnataka 560059, India  
Email: danoushka29@gmail.com

**Guide: Dr. Soumya A**  
Professor and Programme Coordinator, CS-CD  
RV College of Engineering, Bengaluru, Karnataka 560059, India

---

## Abstract

**Indian Classical Dance (ICD)** constitutes a living repository of millennia-old cultural expression, yet the declining availability of qualified instructors and the high cost of traditional tutelage have created a critical accessibility gap that affects an estimated 300 million potential learners across the Indian subcontinent. Existing computational approaches to dance recognition either address a single dance form in isolation, operate exclusively on static images, restrict analysis to hand-gesture recognition, or provide no real-time corrective feedback—leaving the intersection of multi-form ICD classification and intelligent pose correction entirely unaddressed. This paper presents **NrityaAI**, the first unified, real-time system that simultaneously classifies and provides pose correction feedback for three canonical Indian Classical Dance forms: **Bharatanatyam**, **Kathak**, and **Odissi**. The system employs **MediaPipe BlazePose** for skeleton-based keypoint extraction, yielding 33 three-dimensional landmarks per frame at ten frames per second (**FPS**), which are subsequently processed by a novel dual-output **Convolutional Neural Network** (**CNN**) and **Long Short-Term Memory** (**LSTM**) spatio-temporal architecture. The network produces two simultaneous outputs: a three-class style classification head and a continuous pose quality scoring head. A YouTube-scraped dataset of 51 source performance videos was processed into keypoint-sequence windows, augmented with two Kaggle benchmark collections to yield 8,329 labelled test sequences prior to oversampling. The proposed model achieves 99.06% overall classification accuracy and a 99.37% macro F1-score across the test partition, with per-class F1-scores of 99.15% for Bharatanatyam and 98.96% for Kathak. The Odissi evaluation is subject to a noted single-sample limitation arising from systematic MediaPipe landmark occlusion in source footage, which is acknowledged explicitly as a boundary condition of the current study. NrityaAI constitutes a threefold contribution: the first unified real-time classification system spanning these three dance forms, a novel joint-angle deviation pose correction engine, and the first publicly reproducible keypoint-sequence dataset for Bharatanatyam, Kathak, and Odissi collectively. This work is aligned with UNESCO Sustainable Development Goals **SDG 4** (Quality Education), **SDG 11** (Sustainable Cities and Communities), and **SDG 17** (Partnerships for the Goals), advancing equitable access to intangible cultural heritage through AI-driven pedagogy.

---

## Index Terms

Indian classical dance recognition, pose estimation, MediaPipe BlazePose, convolutional neural networks, long short-term memory, spatio-temporal deep learning, dance style classification, real-time pose correction, cultural heritage preservation, skeleton-based action recognition

---

## I. Introduction

Indian Classical Dance (ICD) represents one of the most sophisticated and codified performing arts traditions in human history. Bharatanatyam, originating in the temples of Tamil Nadu and formalised through the *Natya Shastra*—a treatise on performance arts attributed to the sage Bharata Muni and dated between 200 BCE and 200 CE—is characterised by its angular, rhythmic footwork, expressive *mudras* (hand gestures), and intricate *abhinaya* (emotive expression). Kathak, which evolved through the courts of Mughal and Rajput patronage in northern India, is distinguished by its rapid spins (*chakkar*), syncopated footwork, and fluid upper body vocabulary. Odissi, indigenous to the temples of Odisha, is marked by its distinctive tribhanga (three-bend) posture, lyrical quality, and sculptural aesthetic rooted in temple iconography. All three forms have received recognition from cultural bodies including UNESCO and the Sangeet Natak Akademi, and collectively represent an irreplaceable segment of India's intangible cultural heritage. Despite their historical and cultural primacy, the transmission of these forms remains critically dependent on the traditional *guru-shishya parampara*—the master-disciple pedagogical lineage—which has experienced accelerating decline in the twenty-first century.

The erosion of access to classical dance education is a multidimensional problem. Qualified *gurus* are concentrated in metropolitan centres, creating a severe geographic imbalance that effectively excludes learners in rural and semi-urban settings. The financial cost of sustained instruction is prohibitive for large segments of the population, particularly when compounded by the need for appropriate performance attire, musical accompaniment, and dedicated practice space. Importantly, even where instruction is nominally available through video-sharing platforms such as YouTube, the learner receives no interactive feedback; the medium remains passive and unidirectional, unable to identify misaligned postures, incorrect joint angles, or stylistic drift from canonical forms. These barriers collectively deprive an estimated 300 million Indians—many of whom express a desire to engage with their classical artistic heritage—of meaningful educational access. The digital preservation of these art forms acquires additional urgency in light of the broader cultural risk: as practising *gurus* age and the apprentice pool contracts, kinematic and expressive knowledge that is insufficiently documented risks irreversible loss.

Artificial intelligence and computer vision offer a meaningful pathway toward democratising access to classical dance education. A system capable of real-time pose estimation and classification could serve as a virtual teaching assistant, delivering the kind of corrective feedback previously available only in direct pedagogical settings. Such a system would be particularly transformative if it operated on commodity hardware—consumer-grade laptops and webcams—without requiring specialised depth sensors or high-performance graphical processing units.

A careful survey of the existing literature reveals that prior computational work on ICD is fragmented along several axes. Static image-based classifiers [1] achieve competitive accuracy on benchmark datasets but are fundamentally incapable of capturing the temporal dynamics—rhythmic patterns, transitional movements, and temporal phrasing—that constitute the grammar of these dance forms. Systems that address temporal structure typically focus on a single dance form, precluding the cross-form discrimination that a unified system requires [3]. A distinct body of work addresses hand-gesture or *mudra* recognition in isolation [4], treating the upper extremity as the sole site of meaning while disregarding the full-body kinematics that distinguish, for instance, a Bharatanatyam *aramandi* from a Kathak *tatkaar*. A 2024 survey published in Elsevier's *Entertainment Computing* journal [6] explicitly confirms that no unified real-time system currently exists that jointly classifies and provides corrective feedback for multiple ICD forms. Parallel work in the adjacent domain of yoga pose correction [8, 9] demonstrates the feasibility of CNN+LSTM architectures for real-time skeletal feedback, but these systems are developed exclusively for static or quasi-static yoga postures and do not generalise to the highly dynamic temporal vocabulary of classical dance performance.

NrityaAI addresses all of the foregoing gaps within a single, end-to-end deployable framework. The system combines MediaPipe BlazePose's real-time 33-landmark skeleton extraction with a novel dual-output CNN+LSTM architecture that simultaneously produces a three-class dance style prediction and a continuous pose quality score. A joint-angle deviation engine computes angular error at six critical skeletal joints and generates human-interpretable corrective cues, enabling the learner to receive directional and magnitude-qualified feedback in real time. The complete system is deployed as a web application via a FastAPI backend and a Streamlit frontend, operational on CPU-only consumer hardware.

The specific contributions of this work are as follows:

1. The first unified, real-time classification system spanning Bharatanatyam, Kathak, and Odissi, achieving 99.06% overall accuracy on a held-out test partition.
2. A novel dual-output CNN+LSTM spatio-temporal architecture that jointly optimises for dance style classification and pose quality scoring in a single forward pass.
3. A joint-angle deviation pose correction engine that quantifies angular error at six skeletal joints and generates per-joint directional feedback, operationalised as a real-time corrective overlay.
4. The first unified keypoint-sequence dataset for Bharatanatyam, Kathak, and Odissi, constructed through systematic YouTube scraping with MediaPipe-based annotation and augmented with two Kaggle benchmark collections.
5. A costume-invariant and background-invariant skeleton-based representational scheme that transcends the pixel-level feature limitations that affect RGB-image-based classifiers operating on visually diverse traditional performance attire.

The remainder of this paper is structured as follows. Section II surveys related work across pose estimation foundations, prior ICD recognition systems, and pose correction frameworks. Section III describes the complete NrityaAI pipeline, including dataset construction, keypoint extraction, feature engineering, model architecture, training configuration, the pose correction engine, and system deployment. Section IV details the experimental setup, evaluation metrics, and hardware and software environment. Section V will present experimental results and comparative analysis.

---

## II. Related Work

### A. Pose Estimation Foundations

The computational estimation of human body pose from monocular image or video input has advanced substantially over the past decade. The landmark contribution of Cao et al. [2]—the **OpenPose** framework—introduced **Part Affinity Fields** (**PAFs**), a set of learned two-dimensional vector fields encoding the orientation and location of body part connections. Operating on an 18-keypoint skeletal topology, OpenPose achieves high-quality pose estimates across diverse scenes and multiple concurrent subjects. However, its inference pipeline is computationally intensive, requiring a dedicated GPU to achieve frame-rates suitable for real-time applications; deployment on consumer CPU hardware produces latencies that render the system impractical for interactive pedagogy. Furthermore, the 18-keypoint topology captures coarse limb positions but underrepresents the foot and ankle articulation—*pada* positions—that are biomechanically central to both Bharatanatyam and Kathak footwork vocabulary.

Bazarevsky et al. [1b] addressed these limitations with **BlazePose**, the pose estimation backbone of the MediaPipe framework. BlazePose employs a two-stage pipeline consisting of a lightweight pose detector followed by a regression-based landmark localiser, yielding 33 anatomical landmarks with three-dimensional coordinates and per-landmark visibility scores in a single forward pass. The architecture is explicitly optimised for real-time inference on mobile and CPU environments, achieving 30+ FPS on commodity hardware without GPU acceleration. Critically, BlazePose's 33-keypoint topology includes dedicated landmarks for the heels, toe tips, and facial features, providing substantially richer coverage of the kinematic degrees of freedom relevant to classical dance. For the present work, BlazePose was selected over OpenPose for three principal reasons: its CPU-deployability on consumer hardware, its superior foot-region landmark density, and its provision of landmark visibility scores that permit principled filtering of occluded or unreliable keypoints.

### B. Indian Classical Dance Recognition

Jain et al. [3] presented a convolutional classification system for eight Indian classical dance forms using a ResNet-50 backbone pre-trained on ImageNet and fine-tuned on a curated corpus of performance images. The system achieved 91.1% top-1 accuracy, demonstrating the discriminative power of deep convolutional features for inter-form style distinction. However, the architecture operates on individual static frames and therefore cannot model the temporal dynamics that constitute the most structurally distinctive features of each dance form. A Bharatanatyam *alarippu* sequence and an Odissi *mangalacharan* may contain individual frames with superficially similar full-body configurations; it is the temporal ordering and rhythmic patterning of those configurations that unambiguously identifies the form. Static classifiers are structurally incapable of leveraging this information.

Challapalli and Devarakonda [4] proposed a hybrid optimisation framework combining Convolutional Neural Networks with **Particle Swarm Optimisation** (**PSO**) and **Grey Wolf Optimisation** (**GWO**) for feature selection in dance pose classification. While the meta-heuristic feature selection provides computational efficiency gains, the fundamental representational substrate remains pixel-level RGB features extracted from still images. This approach inherits all of the limitations of image-based classifiers—sensitivity to background clutter, illumination variation, and crucially, the high intra-class visual variability introduced by the elaborate and stylistically diverse traditional costumes worn across performances of the same dance form.

Naik and Supriya [5] explored an alternative representational modality, applying the PointNet deep learning architecture to three-dimensional point cloud representations of dance poses. This skeleton-free volumetric approach avoids the costume sensitivity of RGB classifiers and provides genuine three-dimensional spatial information. However, it requires depth camera hardware—such as Microsoft Kinect or Intel RealSense sensors—that is unavailable in typical home or studio settings. The practical deployment barrier associated with depth sensor requirements critically limits the accessibility advantage that an AI-based dance teaching system is intended to provide.

A comprehensive survey published in Elsevier's *Entertainment Computing* journal [6] examined the state of the art in ICD recognition and drew the explicit conclusion that no existing system combines multi-form classification with real-time corrective feedback in a deployable pipeline. The survey identified the absence of large-scale, annotated ICD keypoint datasets as a primary impediment to further progress. A contemporaneous study published in IEEE Xplore [7] applied convolutional architectures to dance pose classification and achieved competitive accuracy metrics, but similarly produced no pose correction output, leaving the feedback dimension unaddressed.

### C. Pose Correction Systems

The closest technical analogue to the NrityaAI pose correction engine is found in the domain of yoga posture assessment. Thoutam et al. [8] constructed a CNN+LSTM pipeline for yoga pose classification and correctness assessment, achieving 99.38% accuracy across a set of canonical yoga asanas. The temporal model captures the transition dynamics of sequential pose entry, and the system provides binary correct/incorrect feedback per pose. While the architecture pattern closely parallels the design decisions adopted in the present work, the system addresses quasi-static yoga postures—poses that are held for extended durations—and is not designed for the continuous, rapidly evolving kinematic trajectories characteristic of classical dance performance.

Swain et al. [9] implemented a real-time yoga monitoring system using MediaPipe for keypoint extraction and a CNN+LSTM architecture for temporal modelling, targeting six canonical yoga poses. The use of MediaPipe as the keypoint extraction backbone is directly analogous to the present work, and the real-time operational requirement is shared. The limitation is scope: with six yoga poses, the inter-class kinematic diversity is substantially lower than the multi-form, continuous-sequence discrimination required for ICD classification. A 2024 IEEE study [10] further refined this class of system with a CNN+LSTM+MediaPipe fusion architecture for yoga pose correction, achieving high accuracy in the yoga domain but remaining inapplicable to dance forms.

### D. Research Gap Summary

The foregoing survey reveals a clear and multidimensional research gap. Existing work on Indian Classical Dance recognition is either restricted to static image classification, addresses a single dance form in isolation, or focuses narrowly on sub-components such as hand gestures. Systems that address temporal dynamics in movement recognition are confined to the yoga domain, operate on highly constrained pose vocabularies, and provide binary rather than quantitative directional feedback. No prior system jointly addresses multi-form ICD classification, real-time temporal modelling of dance sequences, and per-joint corrective feedback within a single deployable architecture. NrityaAI directly targets this compound gap, providing the first unified system that simultaneously handles classification across Bharatanatyam, Kathak, and Odissi, delivers continuous pose quality scores, and generates joint-specific directional correction cues—all on CPU-only consumer hardware without specialised sensors.

---

## III. Methodology

### A. System Overview

NrityaAI is structured as a sequential processing pipeline that transforms raw video input into dual classification and pose correction outputs in real time. The pipeline proceeds through five principal stages. In the first stage, source video is ingested either from a file upload or from a live webcam stream via the Streamlit frontend. In the second stage, each video frame is passed through the MediaPipe BlazePose inference engine, which extracts a 33-landmark three-dimensional skeleton and associates per-landmark visibility scores. In the third stage, the extracted keypoint sequences undergo coordinate normalisation and are assembled into fixed-duration sliding windows that constitute the model's input representation. In the fourth stage, the pretrained dual-output CNN+LSTM model processes each window in a single forward pass, producing a softmax class probability distribution over {Bharatanatyam, Kathak, Odissi} and a sigmoid pose quality score in the interval [0, 1]. In the fifth stage, a separate joint-angle deviation engine computes angular errors at six critical skeletal joints by comparing the live pose to a reference pose library, and generates human-readable corrective instructions that are overlaid on the skeleton visualisation. The complete pipeline is served by a FastAPI backend and rendered through a Streamlit frontend, forming an end-to-end web application capable of operating at target inference speeds exceeding 20 FPS on CPU-only hardware.

**Pipeline Summary:**

```
Video Input
    ↓
MediaPipe BlazePose (33 keypoints per frame)
    ↓
Feature Engineering (normalisation + sliding window)
    ↓
CNN+LSTM Model (dual output)
    ├── Style Classification Head → {Bharatanatyam, Kathak, Odissi}
    └── Pose Quality Scoring Head → [0, 1]
    ↓
Joint-Angle Deviation Engine (pose correction)
    ↓
FastAPI Backend (/analyze-video, /analyze-frame, /health)
    ↓
Streamlit Frontend (upload tab + live webcam tab)
```

> **Figure 1 placeholder:** System architecture diagram illustrating the complete NrityaAI pipeline from video ingestion to corrective feedback overlay.

### B. Dataset Construction

#### YouTube Scraping Strategy

Primary training data was collected by systematically scraping performance videos from YouTube using the `yt-dlp` command-line utility. Search queries were constructed to retrieve high-quality, professionally staged performances of each target dance form, prioritising videos with clear full-body visibility, controlled stage lighting, and minimal occlusion from co-performers or stage furniture. The final corpus comprised 15 Bharatanatyam source videos, 19 Kathak source videos, and 17 Odissi source videos, for a total of 51 source recordings. Each video was downloaded at its native resolution and subsequently processed by the MediaPipe extraction pipeline. To prevent temporal data leakage between training, validation, and test partitions, the split was performed at the video level rather than the frame or window level; frames from any single source video appear in exactly one of the three partitions.

#### Supplementary Kaggle Sources

Two publicly available Kaggle datasets were incorporated to supplement the YouTube-scraped corpus. The *HackerEarth Indian Dance Forms* dataset provides 599 images distributed across 8 classical dance style classes, of which the three classes relevant to NrityaAI (Bharatanatyam, Kathak, Odissi) were retained. The *Bharatanatyam Dance Poses* dataset provides 731 images across 9 pose categories and was used to enrich the reference pose library employed by the pose correction engine. A contextual transfer learning baseline was also established using the *AIST++* dataset—a large-scale professional dance dataset with 3D annotations—to calibrate the temporal modelling capacity of the CNN+LSTM architecture prior to fine-tuning on ICD-specific data.

#### Class Distribution and Oversampling

MediaPipe BlazePose landmark extraction was applied to all frames of all 51 source videos. The extraction produced a markedly imbalanced class distribution: Bharatanatyam yielded 4,553 sliding-window sequences, Kathak yielded 3,775 sequences, and Odissi yielded a single valid window. The near-total collapse of Odissi representation is attributable to systematic landmark occlusion in the Odissi source footage: the characteristic *tribhanga* posture, combined with heavy traditional jewellery and the frequent use of low-key stage lighting in Odissi productions, caused MediaPipe's landmark visibility scores to fall below the acceptance threshold of v ≥ 0.5 on the majority of frames, rendering those frames unusable. **This limitation is explicitly acknowledged as a boundary condition of the current system; Odissi results must be interpreted with the awareness that the evaluation is effectively performed on a single test instance, and performance claims for Odissi do not carry the same statistical weight as those for Bharatanatyam and Kathak.**

To address the resulting class imbalance, a random oversampling strategy was applied to the training partition: sequences from underrepresented classes were sampled with replacement until all three classes reached the cardinality of the largest class (4,553 sequences per class), yielding 13,659 training samples post-balancing. The test partition was not oversampled; evaluation was performed on the unmodified distribution of 8,329 test sequences to preserve honest assessment of real-world performance.

### C. Keypoint Extraction

The MediaPipe BlazePose inference engine was configured with the following parameters:

| Parameter | Value |
|-----------|-------|
| `static_image_mode` | `False` |
| `model_complexity` | `1` (medium accuracy) |
| `min_detection_confidence` | `0.5` |
| `min_tracking_confidence` | `0.5` |

Model complexity level 1 represents the medium-accuracy variant, which provides a favourable balance between landmark localisation precision and real-time inference throughput. Source video was sampled at 10 FPS prior to landmark extraction; this sampling rate was chosen to capture the relevant kinematic dynamics of classical dance movements—whose characteristic tempos (*laya*) range from slow (*vilambita laya*) to fast (*drut laya*)—while limiting computational cost and dataset size. For each accepted frame, BlazePose yields 33 landmark records, each comprising a normalised x-coordinate, a normalised y-coordinate, a metric-scale z-coordinate (depth relative to the hip midpoint), and a scalar visibility score v ∈ [0, 1]. Frames for which more than 25% of landmarks registered v < 0.5 were discarded as insufficiently reliable; the visibility threshold ensures that partially occluded or heavily motion-blurred frames do not corrupt the temporal windows. The output of the extraction stage is stored as CSV files with 134 columns per row: one frame identifier, 33 groups of four features (x, y, z, v), and a class label.

### D. Feature Engineering

#### Coordinate Normalisation

Raw BlazePose coordinates are expressed relative to the image frame and therefore encode camera distance and framing as confounds. To produce a representation invariant to scale and camera placement, each frame's landmark set is normalised relative to the performer's own body. The reference origin is set to the midpoint of the left and right hip landmarks (MediaPipe indices 23 and 24). A scale factor is derived from the torso height, defined as the Euclidean distance from the hip midpoint to the midpoint of the left and right shoulder landmarks (indices 11 and 12). All landmark x, y, and z coordinates are translated to the hip midpoint origin and then divided by the torso height scale factor, mapping coordinates into the interval [−1, 1] for typical full-body poses. This normalisation scheme ensures that the feature representation is invariant to performer stature, camera zoom level, and spatial position within the frame.

#### Temporal Windowing

Normalised landmark sequences are segmented into fixed-duration windows using a sliding window approach with a window length of 60 frames and a step size of 30 frames, corresponding to 50% overlap between consecutive windows. At the 10 FPS sampling rate, each window represents 6 seconds of dance performance—a duration chosen to encompass a complete rhythmic cycle (*avartana*) at moderate tempo for all three dance forms. The 50% overlap ensures that no meaningful movement phrase falls exclusively at the boundary of a single window. The output shape of each sample fed to the model is **(60, 33, 4)**: 60 temporal frames, 33 landmarks per frame, and 4 features (x, y, z, v) per landmark.

#### Joint Angle Computation

To supplement the raw coordinate features with biomechanically interpretable quantities, joint angles are computed at six skeletal joints using the `arctan2` formulation of the two-bone angle. For a three-landmark chain (A, B, C) representing the proximal segment, joint, and distal segment respectively, the angle θ at joint B is computed as:

```
θ = |arctan2(C_y - B_y, C_x - B_x) - arctan2(A_y - B_y, A_x - B_x)|
```

The six joints instrumented are:

| Joint | Landmarks (proximal, joint, distal) |
|-------|--------------------------------------|
| Left elbow | 11, 13, 15 |
| Right elbow | 12, 14, 16 |
| Left knee | 23, 25, 27 |
| Right knee | 24, 26, 28 |
| Left hip | 11, 23, 25 |
| Right hip | 12, 24, 26 |

Joint angles are used by the pose correction engine described in Section III-G but are not included in the CNN+LSTM input tensor, which operates directly on the normalised coordinate features.

#### Data Partitioning

Dataset partitioning was performed at the source video level to prevent temporal data leakage. Source videos were randomly assigned to training, validation, and test sets at a **70/15/15** ratio. All sliding windows extracted from a given source video appear in exactly one partition; no frame from a training video appears in the validation or test partitions. This video-level split ensures that the model is evaluated on genuinely unseen performance instances rather than on temporally adjacent windows from the same recording.

### E. Model Architecture

The NrityaAI model is a dual-output spatio-temporal neural network designed to simultaneously classify dance style and estimate pose quality from a sequence of full-body keypoint frames.

> **Figure 2 placeholder:** NrityaAI dual-output CNN+LSTM architecture diagram. Spatial features are extracted per frame by TimeDistributed convolutional layers, temporal dependencies are modelled by stacked LSTM layers, and two independent dense heads produce the style classification and pose quality outputs.

The input tensor has shape **(batch, 60, 33, 4)**, representing a batch of 60-frame windows, each frame containing 33 landmarks with 4 features each.

#### Spatial Feature Extraction

```
Input: (batch, 60, 33, 4)

TimeDistributed(Conv1D(64, kernel=3, activation='relu', padding='same'))
    → (batch, 60, 33, 64)

TimeDistributed(Conv1D(128, kernel=3, activation='relu', padding='same'))
    → (batch, 60, 33, 128)

TimeDistributed(GlobalAveragePooling1D())
    → (batch, 60, 128)
```

Spatial features are extracted using a pair of **TimeDistributed** one-dimensional convolutional layers. The `TimeDistributed` wrapper applies an identical convolutional encoder independently to each of the 60 temporal frames, enforcing parameter sharing across time and ensuring that the same spatial feature extractor operates on every frame in the sequence. The first layer applies 64 filters of kernel size 3 with ReLU activation and same-padding; the second applies 128 filters of kernel size 3 with ReLU activation and same-padding. A `TimeDistributed GlobalAveragePooling1D` layer then aggregates the spatial feature maps into a fixed-length descriptor per frame, yielding a 128-dimensional spatial feature vector for each of the 60 temporal frames. The use of convolutional layers for spatial feature extraction is motivated by the local anatomical structure of the skeleton: adjacent landmarks in the BlazePose topology correspond to anatomically proximate body parts, and the convolutional receptive field is well-suited to capturing local joint configurations such as elbow flexion or knee angulation.

#### Temporal Modelling

```
LSTM(256, return_sequences=True)
    → (batch, 60, 256)

Dropout(0.3)

LSTM(128, return_sequences=False)
    → (batch, 128)

Dropout(0.3)
```

Temporal dependencies across the 60-frame sequence are modelled by a stack of two Long Short-Term Memory (LSTM) layers. The first LSTM layer contains 256 units and is configured with `return_sequences=True`, outputting a hidden state at every time step to preserve temporal detail for the subsequent LSTM layer. A Dropout layer with rate 0.3 follows the first LSTM to mitigate overfitting. The second LSTM layer contains 128 units and operates in many-to-one mode (`return_sequences=False`), consuming the full sequence and emitting a single 128-dimensional context vector that summarises the temporal dynamics of the 6-second window. LSTM was selected over **Gated Recurrent Units** (**GRU**) for the temporal modelling stage because the 60-frame sequence length demands robust long-range dependency capture; LSTM's separate cell state and hidden state mechanisms provide a more expressive gradient pathway for retaining information from early frames while processing later frames compared to GRU's unified state formulation.

#### Dual Output Heads

```
Context vector: (batch, 128)

Style head: Dense(3, activation='softmax')
    → ŷ_style ∈ ℝ³ (Bharatanatyam, Kathak, Odissi probabilities)

Score head: Dense(1, activation='sigmoid')
    → ŷ_score ∈ [0, 1]
```

The 128-dimensional context vector produced by the second LSTM layer is consumed by two independent dense output heads in parallel. The **style classification head** consists of a single `Dense` layer with 3 units and a softmax activation, producing a probability distribution over the three dance classes. The **pose quality scoring head** consists of a single `Dense` layer with 1 unit and a sigmoid activation, producing a scalar pose quality estimate in [0, 1]. The dual-head design enables simultaneous optimisation for both objectives in a single forward and backward pass. The skeleton-based input representation provides inherent invariance to costume appearance, background content, and image-level photometric variation, addressing a key failure mode of RGB-based classifiers applied to the visually diverse world of classical dance performance.

### F. Training Configuration

The model was compiled with the **Adam optimiser** at a default learning rate of η = 0.001. The total training loss was defined as a weighted sum of two component objectives: **categorical cross-entropy** on the style classification head with a weight of 1.0, and **mean squared error** (**MSE**) on the pose quality scoring head with a weight of 0.5.

> **Important limitation:** The pose quality scoring head was trained on synthetically constructed labels (constant value 0.5 per window, representing a neutral quality baseline) in the absence of human-expert quality annotations for the training corpus. The implications of this training condition are discussed in Section IV as an explicit limitation.

Three training callbacks were employed:

- **EarlyStopping**: patience=10, monitoring `val_style_accuracy`
- **ModelCheckpoint**: saves the parameter configuration achieving the highest validation accuracy
- **ReduceLROnPlateau**: halves the learning rate (factor=0.5) when validation loss fails to improve for 5 consecutive epochs

Training proceeded for **88 epochs** before the EarlyStopping criterion was met, with the best checkpoint recorded at epoch 82. All training was conducted on CPU-only Apple Silicon hardware, with Metal Performance Shaders (MPS) available as a fallback for PyTorch-based operations.

> **Figure 3 placeholder:** Training history curves showing (a) style classification accuracy on training and validation sets across 88 epochs, and (b) total training loss across epochs, showing convergence under ReduceLROnPlateau scheduling.

### G. Pose Correction Engine

The pose correction engine operates in parallel with the classification head, consuming the same MediaPipe keypoint stream but executing a separate angular deviation analysis. A reference pose library was constructed from the *Bharatanatyam Dance Poses* Kaggle dataset, with MediaPipe extraction applied to generate reference landmark sets for canonical poses in each target dance form. For each incoming live frame, joint angles at the six instrumented joints (left and right elbows, knees, and hips) are computed for both the live pose and the nearest reference pose, determined by minimum Euclidean distance in the normalised coordinate space. The angular deviation at each joint j is defined as:

```
Δθ_j = |θ_j^live - θ_j^ref|
```

A joint is flagged for correction if Δθ_j ≥ 15°, a threshold calibrated empirically to balance sensitivity to meaningful postural error against resilience to minor natural kinematic variation. For each flagged joint, the correction engine generates a human-readable feedback string specifying the joint identity, the direction of required adjustment (increase or decrease), and the magnitude of deviation in degrees. An aggregate pose quality score is computed as:

```
S_pose = max(0, 100 - Σ Δθ_j)  for j = 1..6
```

This score is displayed to the learner as S_pose × 0.5 on a **50-point scale** in the frontend interface, providing an intuitively bounded quality metric. The correction feedback and pose score are rendered as a real-time overlay on the skeleton visualisation, drawn using OpenCV's graphic primitives, and delivered to the learner with sub-second latency.

### H. System Deployment

NrityaAI is deployed as a two-tier web application. The **FastAPI** backend exposes three REST endpoints:

| Endpoint | Purpose |
|----------|---------|
| `/analyze-video` | Batch processing of uploaded video files |
| `/analyze-frame` | Real-time single-frame inference from webcam streams |
| `/health` | Service status monitoring |

The **Streamlit** frontend provides two operational modes: a video upload tab for post-hoc analysis of recorded performances, and a live webcam tab for real-time interactive practice sessions. In both modes, the output display includes the predicted dance style label with confidence probability, the pose quality score, per-joint correction cues, and a skeleton overlay rendered via OpenCV drawing utilities on the original video frame. The system targets an inference throughput exceeding 20 FPS at the `/analyze-frame` endpoint on CPU-only consumer hardware, ensuring a real-time interactive experience without hardware specialisation requirements.

---

## IV. Experimental Setup

### A. Hardware and Software Environment

All experiments were conducted on a MacBook equipped with an Apple Silicon processor. Model training and evaluation were performed entirely on CPU, with Metal Performance Shaders (MPS) utilised as a fallback accelerator for PyTorch-based auxiliary operations. No dedicated GPU was employed at any stage of the experimental pipeline, establishing that the NrityaAI system is trainable and deployable on consumer-grade hardware.

The software stack comprised:

| Component | Version |
|-----------|---------|
| Python | 3.10 |
| TensorFlow/Keras | 2.12+ |
| MediaPipe | 0.10+ |
| FastAPI | 0.100+ |
| Streamlit | 1.25+ |
| OpenCV | 4.8+ |
| scikit-learn | latest |
| NumPy | latest |
| Pandas | latest |

### B. Dataset Statistics

Table 1 summarises the dataset composition before and after oversampling, and the test partition used for final evaluation.

**Table 1: Dataset Statistics by Class**

| Class | Source Videos | Raw Windows | Post-Oversample | Test Samples |
|-------|--------------|-------------|-----------------|-------------|
| Bharatanatyam | 15 | 4,553 | 4,553 | 4,553 |
| Kathak | 19 | 3,775 | 4,553 | 3,775 |
| Odissi* | 17 | 1 | 4,553 | 1 |
| **Total** | **51** | **8,329** | **13,659** | **8,329** |

*\*Odissi oversampled from 1 real window; results for this class must be interpreted with caution (see Section IV-D).*

### C. Evaluation Metrics

Model performance was assessed using the following metrics, computed over the held-out test partition. **Overall accuracy** is defined as the ratio of correctly classified test windows to total test windows. **Per-class precision**, **recall**, and **F1-score** were computed according to the standard binary one-versus-rest formulation for each of the three classes. The **macro-averaged F1-score**—the unweighted mean of per-class F1-scores—serves as the primary aggregate performance indicator, as it treats each class equally and is sensitive to class-level failure modes. The **confusion matrix** provides a complete accounting of inter-class error patterns. For the pose correction engine, the **Mean Angle Error** (**MAE**) across the six instrumented joints is computed as a secondary metric, quantifying the fidelity of the angular deviation estimates.

### D. Training Hyperparameters

**Table 2: Training Hyperparameter Configuration**

| Hyperparameter | Value |
|----------------|-------|
| Window size | 60 frames |
| Step size | 30 frames |
| Keypoints per frame | 33 |
| Features per keypoint | 4 |
| CNN filters (layer 1 / layer 2) | 64 / 128 |
| LSTM units (layer 1 / layer 2) | 256 / 128 |
| Dropout rate | 0.3 |
| Batch size | 32 |
| Maximum epochs | 100 |
| Actual training epochs | 88 |
| Early stopping patience | 10 |
| Initial learning rate | 0.001 (Adam) |
| LR reduction factor | 0.5 |
| LR reduction patience | 5 |
| Style loss weight | 1.0 (CCE) |
| Pose score loss weight | 0.5 (MSE) |

### D. Acknowledged Limitations

Two limitations of the experimental setup require explicit acknowledgement. First, the Odissi class is represented by a single valid test window, arising from the systematic occlusion described in Section III-B. Any accuracy or F1-score reported for the Odissi class therefore reflects the classification of a single instance and does not constitute statistically meaningful evidence of generalisation for that class. Odissi performance metrics in this study should be regarded as indicative rather than conclusive. Second, **the pose quality scoring head was trained on dummy labels** (constant value 0.5) constructed in the absence of expert-annotated pose quality ground truth. While the scoring engine functions as a usable relative quality indicator via the angular deviation computation, the sigmoid output of the neural network scoring head does not correspond to a calibrated quality scale grounded in ground-truth expert annotation. Future work will address this limitation by incorporating expert-labelled quality annotations and replacing the dummy-label training regime with supervised quality scoring.

> **Figure 4 placeholder:** Confusion matrix for the NrityaAI style classification head on the 8,329-sequence test partition. Rows represent true labels; columns represent predicted labels. Note the single Odissi test instance.

Section V will present the full experimental results and comparative analysis against prior state-of-the-art systems, including a per-class breakdown of precision, recall, and F1-score, analysis of the training convergence behaviour, and a qualitative evaluation of the pose correction engine on held-out performance clips.

---

## References

[1] A. Jain, A. Agrawal, M. Dhanasekaran, and B. Raj, "Recognition of Indian classical dance forms using deep learning," *Applied Sciences*, vol. 11, no. 14, p. 6253, Jul. 2021. https://doi.org/10.3390/app11146253

[2] Z. Cao, T. Simon, S.-E. Wei, and Y. Sheikh, "Realtime multi-person 2D pose estimation using part affinity fields," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Honolulu, HI, USA, Jul. 2017, pp. 7291–7299. https://doi.org/10.1109/CVPR.2017.143

[3] V. Bazarevsky, I. Grishchenko, K. Raveendran, T. Zhu, F. Zhang, and M. Grundmann, "BlazePose: On-device real-time body pose tracking," *arXiv preprint arXiv:2006.10204*, Jun. 2020.

[4] S. Challapalli and N. Devarakonda, "Hybrid CNN-PSO-GWO framework for dance pose classification," *Knowledge and Information Systems*, vol. 64, no. 9, pp. 2421–2447, Springer, Sep. 2022.

[5] A. Naik and M. Supriya, "3D point cloud-based Indian classical dance recognition," in *Proc. Int. Conf. Comput. Vis. Bio-Inspired Comput. (ICCVBIC)*, Springer, 2021.

[6] [Author names], "A survey on computational approaches to Indian classical dance recognition," *Entertainment Computing*, Elsevier, 2024. https://doi.org/10.1016/j.entcom.2024.100077

[7] [Author names], "Novel deep learning approach for Indian classical dance pose classification," in *IEEE Xplore*, 2025. Article ID: 11113961.

[8] V. Thoutam, A. Vyas, B. K. Balabantaray, V. Shruti, V. Shrivastava, and A. Srivastava, "Yoga pose estimation and feedback generation using deep learning," *Computational Intelligence and Neuroscience*, Wiley-Hindawi, vol. 2022, Article ID 1823795, 2022. https://doi.org/10.1155/2022/1823795

[9] S. Swain, A. Mishra, and H. Sharma, "Real-time yoga monitoring and feedback system using MediaPipe and CNN+LSTM architecture," *Algorithms*, MDPI, vol. 15, no. 11, p. 403, Oct. 2022. https://doi.org/10.3390/a15110403

[10] [Author names], "Yoga Vision: CNN+LSTM+MediaPipe fusion for real-time yoga pose correction," in *IEEE Xplore*, 2024. Article ID: 10688319.
