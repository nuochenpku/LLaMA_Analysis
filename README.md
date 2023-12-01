# LLaMA_Analysis
This is official project in our paper: Beyond Surface: Probing LLaMA Across Scales and Layers

## Overview

This projects presents an in-depth analysis of Large Language Models (LLMs), focusing on LLaMA, a prominent open-source foundational model in natural language processing. 
Instead of assessing LLaMA through its generative output, we design multiple-choice tasks to probe its intrinsic understanding in high-order tasks such as reasoning and computation. We examine the model horizontally, comparing different sizes, and vertically, assessing different layers.

We probe the LLaMA models in five high-order tasks:

- **Calculation**
- **Math problem solving (MPS)**
- **Logical reasoning**
- **Truthfulness**
- **Factual knowledge detection**
- **Cross-lingual Reasoning**

We unveil several key and uncommon findings based on the designed probing tasks: 

-  Horizontally, enlarging model sizes almost could not automatically impart additional knowledge or computational prowess. Instead, it can enhance reasoning abilities, especially in math problem solving, and helps reduce hallucinations, but only beyond certain size thresholds;
    
-  In vertical analysis, the lower layers of LLaMA lack substantial arithmetic and factual knowledge, showcasing logical thinking, multilingual and recognitive abilities, with top layers housing most computational power and real-world knowledge.

We expect these findings provide new observations into LLaMA's capabilities, offering insights into the current state of LLMs.


## Results

### Overall Probing 

<p align="center">
  <img src="figure/who_compare.png" alt="Description of image">
</p>

### 7B Probing 

<p align="center">
  <img src="figure/7b_whole.png" alt="Description of image">
</p>

### 13B Probing 

<p align="center">
  <img src="figure/13b_whole.png" alt="Description of image">
</p>


### 70B Probing 

<p align="center">
  <img src="figure/70b_whole.png" alt="Description of image">
</p>

### Calculation Probing

### 1-2Bit
<p align="center">
  <img src="figure/1_2bit_cal.png" alt="Description of image">
</p>

### 3-4Bit
<p align="center">
  <img src="figure/3_4bit_cal.png" alt="Description of image">
</p>

### 5-6Bit
<p align="center">
  <img src="figure/5_6bit_cal.png" alt="Description of image">
</p>

### Cross-lingual Probing

### XMPS-Rea
<p align="center">
  <img src="figure/xmps-rea.png" alt="Description of image">
</p>

### XMPS-Cal
<p align="center">
  <img src="figure/xmps-cal.png" alt="Description of image">
</p>






