# BlueBagEdgeAI ♻️🤖

A weekend hobby project blending **Edge AI**, **IoT**, and **recycling**, inspired by Luxembourg’s blue ValorLux recycling bags 🇱🇺.  

BlueBagEdgeAI uses an **ESP32-CAM** and **YOLOv11n** to run real-time object detection **locally**, no cloud required. Perfect for learning, tinkering, and exploring AI at the edge.

---

## Features

- Detect recyclable items in real-time
- Edge AI inference on low-power hardware
- Modular and simple setup for experimentation
- Open source: tweak, learn, or expand  

---

## Setups 
 - Computer: [README.md](https://github.com/scardoso-lu/BlueBagEdgeAI/blob/main/computer/README.md)
 - Raspberry PI: coming...

## Camera Setup (all envs)

For a step-by-step guide on how to upload code to the ESP32-CAM using Arduino IDE, watch a YouTube tutorial.

Long video: 

[How to setup and use ESP32 Cam with Micro USB WiFi Camera](https://www.youtube.com/watch?v=RCtVxZnjPmY)

Contain links to buy the ESP32-CAM on amazon.

Short video:

[Install the ESP32 Board in Arduino IDE in less than 1 minute (Windows, Mac OS X, and Linux)](https://youtu.be/mBaS3YnqDaU?si=v1chAg5eljwWvHd4)



### This will show:

- How to connect the ESP32-CAM (USB-TTL or ESP32-CAM-MB)

- How to put the board in upload mode

- How to flash the CameraWebServer example

### 📸 Camera Cable Warning (READ THIS)

If the camera does not work, check the flat cable first.

- The cable must be fully inserted

- The blue side of the cable must face the Wi-Fi antenna

- A loose or reversed cable will cause camera errors (for example: 0x106)

👉 Most ESP32-CAM camera problems are caused by a badly connected cable.
