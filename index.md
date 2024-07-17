# Custom Object Detection System
A project that can detect objects, using TensorFlow Lite on the Raspberry Pi, adding convienence to your building experience. 


| **Engineer** | **School** | **Area of Interest** | **Grade** |
|:--:|:--:|:--:|:--:|
| Alex N. | Fremont | Mechanical Engineering | Incoming Junior


![Alexnicholls](https://github.com/user-attachments/assets/769afe56-cca0-4e10-9b00-bdf3642b244a)

# Modification

<iframe width="560" height="315" src="https://www.youtube.com/embed/nfa1NcvNvV8?si=BekNDqx8RI_Do7dC" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

## Summary

For my Modification I made it track multiple objects and I was able to do that with OpenCV and python. In the software I added a feature where the user can choose specifically which objects to detect like mouse or person and it will only detect that object. I thought it would be cool if the camera could follow the objects in real time, so I built a pan tilt servo display, and with w, a, s, and d you can move the servos 180 degrees in 4 directions.

## Challenges

One of the biggest challenges with my modification was making the servos work, I had to give them the right amount of power to them with a battery pack. I wired up the servos with a transistors to reroute the power from the battery pack, I wired them incorrectly and the ground from the battery pack went straight to the raspberry pi. When I plugged it in I fried my Pi for the second time, and I had to get another one. 

## What's Next

If I were to continue this project with more time, I would spent more time understanding AI and adding layers to the pre-trained AI models, and making the servos spin in 360 degrees so it can see the entire room. 






  
# Final Milestone

<iframe width="560" height="315" src="https://www.youtube.com/embed/EV1cqBGiJh8?si=Ei2IFFhwSxzjn1fA" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

## Summary

In my third and final milestone, I developed an object detection software using TensorFlow Lite, enabling real-time object detection on the Raspberry Pi. Leveraging Google's TensorFlow GitHub library, I utilized their optimized computer vision model to achieve swift and efficient object detection and computer vision capabilities on the Raspberry Pi.

## Progress

I followed a detailed tutorial to set up TensorFlow 2 on my Raspberry Pi, which included installing essential Python packages, the speech output package, and the rpi-vision package. I successfully cloned and installed the Adafruit fork of the rpi-vision program and installed TensorFlow 2.15.0. After rebooting the Raspberry Pi, I ran the graphic labeling demo. The system was able to display the objects it detected on screen. The camera correctly identified several items, such as coffee mugs and animals, demonstrating that the object detection setup is working as intended.

## Challenges

One of the biggest challenges with this final milestone was trying to run the command that would start the detection software. The terminal would say that it’s missing a module, I would install that module using a PIP command and it would say it’s missing a different module. For me to get it to work, I had to set up the virtual environment so all of my packages were in one place.

## What's Next

Going forward, some modificaitons that I would make to this project were probably working on the ability to track multiple objects at once and putting a box around the objects that it's detecting.

# Second Milestone

<iframe width="560" height="315" src="https://www.youtube.com/embed/muLqT1Jm2Yo?si=g8nGha933mwRVHtQ" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

## Summary

Since my previous milestone, I have worked on setting up and testing the camera module and the Adafruit Braincraft HAT for my TensorFlow Lite Object Detection project.

## Progress

I Properly connected and configured the Raspberry Pi Camera Module 3 to ensure it captures images correctly. Integrated the Adafruit Braincraft HAT to display the terminal output of the Raspberry Pi.

## Challenges

I encountered challenges in implementing the video feed from the camera to the Adafruit Braincraft HAT. As a solution, I utilized the main monitor on my computer, which ultimately was beneficial due to its better resolution.

## What's Next

Moving ahead, my focus will be on installing TensorFlow 2 and RPI Vision libraries on the Raspberry Pi, followed by testing the object detection capabilities.


# First Milestone

<iframe width="560" height="315" src="https://www.youtube.com/embed/p8tFRpc52To?si=-ktnJg8QgnNMPHI5" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

## Summary

My project is Raspberry Pi object detection through machine learning. My first milestone involved building the hardware. The camera module will take a picture and run it through the tensorflow lite python interpreter and that will return what it thinks it sees.

## Progress

I set up all of the hardware for this project, which includes putting the SD card with Raspberry pi OS into the Raspberry Pi, using the gpio ribbon wire to connect the Adafruit Braincraft HAT to the Raspberry Pi, screwing in the Pi-fan on the back of the Adafruit Braincraft HAT, and plugging in the Raspberry Pi Camera Module 3.  

## Challenges

While building my hardware I accidentally touched the input and output pins which caused the raspberry pi to short circuit and I had to wait for claudia to order a new Raspberry pi, which made building the hardware a little difficult. 

## What's Next

Going further, I plan to start work on the first camera test, and getting the Adafruit Braincraft HAT to show what the camera is seeing. 

# Schematics 

<img width="495" alt="Screenshot 2024-07-09 at 1 05 38 PM" src="https://github.com/VegitarianCalc/Alex_VeryReallyGoodPortfolio_BlueStamp/assets/143454271/2aed104b-b061-4afd-bf0e-1d9c497a2063">


# Code

```c++
import os
import cv2
from picamera2 import Picamera2
import numpy as np


# Initialize Picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()

# Load class names
classNames = []
classFile = "/home/alexnicholls/Desktop/Object_Detection_Files/coco.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

# Load model configuration and weights
configPath = "/home/alexnicholls/Desktop/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "/home/alexnicholls/Desktop/Object_Detection_Files/frozen_inference_graph.pb"
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img, confThreshold=thres,
nmsThreshold=nms)
    if len(objects) == 0:
        objects = classNames
    objectInfo = []
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(),
confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box, className])
                if draw:
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, classNames[classId - 1].upper(),
(box[0] + 10, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, str(round(confidence * 100, 2)),
(box[0] + 200, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    return img, objectInfo

if __name__ == "__main__":
    try:
        while True:
            # Capture frame-by-frame
            frame = picam2.capture_array()

            # Convert the frame from 4 channels to 3 channels
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # Perform object detection
            result, objectInfo = getObjects(frame, 0.45, 0.2)

            # Display the resulting frame
            cv2.imshow('Output', result)

            # Press 'q' on the keyboard to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # When everything done, release the capture and close windows
        picam2.stop()
        cv2.destroyAllWindows()
```

# Bill of Materials

| **Part** | **Note** | **Price** | **Link** |
|:--:|:--:|:--:|:--:|
| Raspberry Pi 4 Model B | Is a small computer that does most of the computing | $61.75 | <a href="https://www.amazon.com/Arduino-A000066-ARDUINO-UNO-R3/dp/B008GRTSV6/](https://www.amazon.com/Raspberry-Model-2019-Quad-Bluetooth/dp/B07TC2BK1X/ref=asc_df_B07TD42S27/?tag=&linkCode=df0&hvadid=380013417597&hvpos=&hvnetw=g&hvrand=7380946922219675202&hvpone=&hvptwo=&hvqmt=&hvdev=c&hvdvcmdl=&hvlocint=&hvlocphy=9032183&hvtargid=pla-774661502856&ref=&adgrpid=77922879259&th=1"> Link </a> |
| Raspberry Pi Camera Module 3 | Is a small modular camera that plugs into the Raspberry Pi | $45.99 | <a href="[https://www.amazon.com/Arduino-A000066-ARDUINO-UNO-R3/dp/B008GRTSV6/](https://www.amazon.com/Arducam-Raspberry-Autofocus-Acrylic-15-22pin/dp/B0BX6N6V98/ref=sr_1_6?crid=2980RI5X7ETHI&dib=eyJ2IjoiMSJ9.LXcwA6pudfrqywsACAjxJ7XKRrNDWZEmJ7FVxfkqHCOqBtozsEshHVIJ5XdqMqhPmc3gUlmSCe_JNC0fEamH45j-KJIPiQVIsP6jrRXP6h_0mwIBZhzF3hoVxVkxhLDgH1BiXxLSG68uYWWk1waYl5KOgSiniN5wRLGyQfGDPgoS7r8qG5b0Fst4YYKsZGBiU5JY3456z43HEtX7tc-2_cRuJ391XvqJkK89wwiN9YY.GRL30M8F2oOmWE0Xy6EKVaNKzfxgHs1NjK3cL1P88fA&dib_tag=se&keywords=raspberry%2Bpi%2Bcamera%2Bmodule%2B3&qid=1720631713&sprefix=raspberry%2Bpi%2Bcamera%2Bmodule%2B3%2Caps%2C173&sr=8-6&th=1)"> Link </a> |
| Adafruit Braincraft HAT | Is a display for Raspberry Pi | $44.95 | <a href="https://www.adafruit.com/product/4374"> Link </a> |
| HDMI to USB converter | Converts HDMI feed from Raspberry Pi to USB | $14.99 | <a href="https://www.amazon.com/Capture-Streaming-Broadcasting-Conference-Teaching/dp/B09FLN63B3/ref=asc_df_B09FLN63B3/"> Link </a> |
| Mini HDMI to HDMI | Converts Mini HDMI feed to HDMI | $8.49 | <a href="[https://www.amazon.com/Capture-Streaming-Broadcasting-Conference-Teaching/dp/B09FLN63B3/ref=asc_df_B09FLN63B3/](https://www.amazon.com/Amazon-Basics-Speed-Source-Display/dp/B014I8UEGY/ref=sr_1_1_ffob_sspa?crid=1LQLRCSP3JTFZ&dib=eyJ2IjoiMSJ9.0WyHRjCGMuzEGORjc5fjZfne5mFWRoXQ265wyEAbngY1SCMdivgQh-uJOLI-OuPYLhebjV1elOZabnbWkEmKBfnZ8N054xTHTEEvtcFFtREwseRyhq0dWPXsilPxNosJy13JWp64JhXC5kvDXWKcGfvImNVHbGAsygMotAcVicDGoaxPZSHw63vrNs7D5I1DTfJMtjxdZjD7lEe2hu41Ax2s9jELF3rY9AMNNXyIHvPeCxy1puLZOSdiyJMcFDzvIGAd8cATwdBUJaoH_NtaYvtBkKt1_JLPfbrDrdjgJmY.K7j5uwzRBxDwguYL90pdYMRX-NP12wjl9YUxX48rP6g&dib_tag=se&keywords=hdmi%2Bto%2Bmini%2Bhdmi&qid=1720632717&s=electronics&sprefix=HDMI%2Bto%2Bmini%2Celectronics%2C178&sr=1-1-spons&sp_csd=d2lkZ2V0TmFtZT1zcF9hdGY&th=1)"> Link </a> |


# Starter Project 

<iframe width="560" height="315" src="https://www.youtube.com/embed/dIFbhf59PQw?si=9BH6f1EvKj84Bn3K" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>


This is a retro game arcade, you can play SNES tetris with the up, down, left, right buttons and rotational knob, it has a screen to display the score and the blocks alongside a START and STOP button.
- Put simply, all of the input buttons and controls that the player can use will be sent to the micro controller chip in the center and will output movement on the LED display, sound in the peizo buzzer, and the score seen on the top right.
- Something that was particularly challenging was nailing down the soldering, putting the right amount soldering wire on the circuit and lining it up just straight was hard for me. 
- I have made technical advances in understanding how soldering plays a role into assembly of circuits and projects.
