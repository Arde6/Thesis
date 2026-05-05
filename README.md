## Thesis
Artturi Juurenheimo

# Neural Networks Inference on Different Computing Platforms

The objective of this thesis was to compare the performance and cost-effectiveness of neural network inference across three differenct computing platforms: ESP32-S3, Raspberry Pi 5 and a general pc / a server / a cloud service. This thesis was made as a continuation of Atte Mäki-Kerttulas Thesis found at: https://github.com/AtteMK/Thesis-work and https://urn.fi/URN:NBN:fi:amk-2025120934201. Credits of the neural network trainer under pytorch-project folder go to him. My thesis can be found at: *I will add the link here when I have it*

## esp-inferring

This is an esp-idf project made for **Waveshare esp32-s3 pico** board.

### IMPORTANT NOTE
- You need to enable the spi psram and flash memory size in ```idf.py menuconfig```

## pytorch-project

This is a modified version of Attes pytorch trainer. Original code was modified so that it outputs the scaler as well and I also added the ```quantizer.py```, ```inference.py```, ```converter.py``` and ```bme688_simulator.py```