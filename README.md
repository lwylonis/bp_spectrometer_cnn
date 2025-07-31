# FPGA Spectrometer CNN Inference (TAPA HLS)

> FPGA Implementation of CNN-based spectrometer inversion model using TAPA HLS.

---

## Table of Contents

- [Overview](#overview)  
- [Repository Structure](#repository-structure)  
- [Prerequisites](#prerequisites)  
- [Getting Started](#getting-started)  
- [Build & Run](#build--run)  
- [Testing & Verification](#testing--verification)  
- [Performance & Resource Utilization](#performance--resource-utilization)  
- [Contributing](#contributing)  
- [License](#license)  
- [References](#references)

---

## Overview

This repo implements the trained CNN from [ml-repo-name](https://github.com/lwylonis/spectrometer_ml) on an FPGA platform using TAPA HLS.  
It demonstrates real-time spectrometer inversion with Xilinx devices, achieving low latency and high throughput.

---

## Repository Structure

```text

├── cnn/  
│   ├── data/              # model weights and I/O binaries  
│   │   ├── conv1_weight.bin  
│   │   ├── bn1_weight.bin  
│   │   ├── fc2_weight.bin  
│   │   ├── input.bin  
│   │   └── output.bin  
│   ├── include/           # header file
│   │   └── cnn.h
│   ├── Makefile  
│   └── src/  
│       ├── cnn.cpp        # TAPA kernel
│       ├── host.cpp       # functions used by host
│       └── main.cpp       # performance benchmark/verification
├── epoch050.pth           # trained PyTorch checkpoint  
├── LICENSE  
├── README.md              # this file  
└── scripts/  
    └── pth_to_bin.py      # helper to convert .pth → .bin 

```

## Prerequisites

- **Hardware**: Xilinx [INSERT FPGA HERE] (or modify for your board)  
- **Software**:  
  - TAPA HLS v0.1.20250623
  - Xilinx Vivado/Vitis 2023.2  
  - Python 3.8+ with dependencies (see `host/requirements.txt`)  

## Getting Started