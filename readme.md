# Juggler Scheduler


## Overview

Juggler Scheduler is an intelligent task scheduling system optimized for machine learning inference on edge devices. It dynamically balances workloads across resource-constrained environments using a RabbitMQ-based client-consumer architecture.

## Key Features

- **Distributed Scheduling**: Efficiently distribute ML inference tasks across multiple edge devices according to Power efficiency and Accuracy of devices and models.
- **Resource-Aware**: Considers CPU, memory, and specialized hardware availability like GPU, DLAs
- **Fault Tolerance**: Gracefully handles node failures and network interruptions
- **Scalability**: Easily scale to accommodate additional edge devices
- **Low Overhead**: Minimal resource consumption for the scheduler itself
- **Heterogeniety**: Support for heterogeneous edge device deployments

## Architecture

```
                     ┌─────────────────┐
                     │                 │
                     │   Controller    │
                     │                 │
                     └─────────┬───────┘
                               │
                               │
                     ┌─────────▼───────┐
                     │                 │
                     │  RabbitMQ based │
                     │     Juggler     │
                     └─────────┬───────┘
                               │
             ┌─────────────────┼─────────────────┐
             │                 │                 │
     ┌───────▼──────┐  ┌───────▼──────┐  ┌───────▼──────┐
     │              │  │              │  │              │
     │ Edge Device  │  │ Edge Device  │  │ Edge Device  │
     │  Consumer    │  │  Consumer    │  │  Consumer    │
     │              │  │              │  │              │
     └──────────────┘  └──────────────┘  └──────────────┘
```

## Project Overview
Juggler Scheduler is an intelligent task scheduling system designed specifically for edge computing environments. This project implements a distributed scheduling framework that optimizes the execution of machine learning inference workloads across resource-constrained edge devices.
## Key Features

Dynamic workload balancing across edge devices
RabbitMQ-based client-consumer architecture for reliable message passing
Performance-aware scheduling based on device capabilities
Support for heterogeneous edge device deployments
Adaptive resource allocation for ML model inference


## Performance Evaluation
We conducted comprehensive performance metrics testing on-premise with edge devices using five different ML models: ResNet18, ResNet50, ResNext50x32, MobileNetv3, Squeezenet

Evaluated inference latency, throughput, and energy consumption
Tested across varying device specifications and network conditions
Compared scheduling algorithms under different load patterns
Identified optimal configurations for different deployment scenarios

The initial benchmarking results guided the development of our scheduling algorithms, demonstrating significant improvements in overall system efficiency compared to traditional approaches.

ORIN NANO
orin
ssh orin@192.168.50.135
sudo docker run --runtime nvidia -it --rm -v /run/jtop.sock:/run/jtop.sock --network=host dustynv/l4t-pytorch:r36.4.0
sudo docker exec -it 7b6a bash

Jetson NANO
enigma123
ssh newcastleuni@192.168.50.94
sudo docker run --runtime nvidia -it --rm -v /run/jtop.sock:/run/jtop.sock --network=host dustynv/l4t-pytorch:r32.7.1
sudo docker exec -it 7b6a bash

scp -r images newcastleuni@192.168.50.94:nvdli-data
scp imagenet-classes.txt newcastleuni@192.168.50.94:nvdli-data
sudo docker cp imagenet-classes.txt 8a84:/home


Raspberry Pi
ssh pi@192.168.50.203
raspberry
