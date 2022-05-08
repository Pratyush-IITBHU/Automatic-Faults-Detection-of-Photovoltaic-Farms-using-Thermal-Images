<div>
<img src=
"/assets/logo_eee.png" 
         alt="Dronetech logo" 
         align="right" 
         width="200">
<img src=
"/assets/IITBHU-Logo.jpg" 
         alt="technex logo" 
         align="left" 
         width="90">
</div>
<br><br><br>

<h1 align="center"><ins> Automatic Faults Detection of Photovoltaic Farms using Thermal Images</ins> </h1>
<br>

<div align="center">

<img src= "/assets/single_detect.png" alt="Drone" align="left" width="230">

<img src= "/assets/a89.png" alt="Drone" align="right" width="230">

<img src= "/assets/a83.png" alt="Drone" align="center" width="230">

</div>
<br>
<h4 align="center">
<b> <INS>EXPLORATORY PROJECT </INS>  </b>
</h4>

<h6 align="center">
<b> UNDER THE GUIDANCE OF PROFESSOR R.K. PANDEY
  </b>
</h6>

##### TEAM MEMBERS:

- <B>PRATYUSH SINGH</B>
- <B>CHAHAK JETHANI</B>
- <B>OM PRAKASH</B>

<div align="center">
<img src= "/assets/drone2.jpg" alt="Drone" align="center" width="490">
</div>

## Table of contents

  <ol>
    <li>
       <a href="#abstract">Abstract</a>
    </li>
    <li>
      <a href="#introduction">Introduction</a>
    </li>
    <li>
     <a href="#model and algorithm"> Model and Algorithm</a>
     <ul>
        <li><a href="#yolov5">YOLOv5</a></li>
        <li><a href="#dataset preparations">Data Preparations</a></li>
        <li>
         <a href="#predictions made">Predictions Made</a>
          <ul>
           <li><a href="#webots-simulator">PV array detection</a></li>
           <li><a href="#important-note">Single PV module detection</a></li>
           <li><a href="#webots-simulator">PV module with fault</a></li>
         </ul>
        </li>
        <li><a href="#setup-requirements">Model Setup Requirements</a></li>
        <li><a href="#initialisation">Initialisation</a></li>
    </ul>
   </li>
    <li><a href="#scoring">Types of Faults</a></li>
    <li><a href="#instructions-regarding-submissions">Results and Prediction scores</a></li>
  </ol>

# Abstract
  - Because of increasing demand of renewable energy, proper management of photovoltaic hubs is important. But fault detections in them is still a challenge. The main temperature consequenced faults can be detected using Thermal analysis of PV modules. In this paper we used **YOLOv5 (YOU LOOK ONLY ONCE version 5)** for detections, which is a novel state of art convolutional neural network (CNN) that detects objects in real-time with great accuracy. This approach uses a single neural network to process the entire picture, then separates it into parts and predicts bounding boxes and probabilities for each component. Using YOLOv5 we detected:
  <br>
      - **Photovoltaic array module**
      - **Single PV module**
      - **Faulted PV module (on the basis of temperature variation)** 
  <br>

<div align = "center">
 <img src= "/assets/quadcopter.png" align = "left" width = 350>
 <img src= "/assets/thermal.png" align = "right" width = 360>
</div>
<div align = "left">
<br><br><br>
<br><br>
</div>
<br>
<p>.....................</p>

- This model will be used in UAV quadcopter for detections of faults in our campus. It will be uploaded using Ardu-mission planner in the Pixhawk and jetson-nano will be used as companion computer.

# Introduction

<p align="justify">Today renewable energy sources are revolutionising the sector energy generation and represents the only alternative to limit fossil fuel usage. Because of its increasing needs, this sector of renewables energy generation is to be managed in the way so as to fulfil the ever-increasing requirements and demands. The most common and widely used renewable source is photovoltaic (PV) power plants. Which is one of the main systems adopted to produce clean energy. Monitoring the state of health of its system machines and ensuring their proper working is essential for keeping maximum efficiency. However, due to tremendously large area of solar farms, present techniques are time demanding, cause stops to the energy generation, and often require laboratory instrumentation, thus being not cost-effective for frequent inspections. Moreover, PV plants are often located in inaccessible places, making any intervention dangerous.</p>

<p align="justify">So automated fault and error detections in solar plants is today among the most active subjects of research. In this paper, we have used YOLOv5 deep learning network for detections of solar panels and faults in thermal images of solar farm. </p>

<p align = "justify">Since it is known that photovoltaic modules consist of PV cell circuits sealed in an environmentally protective laminate and are the fundamental building blocks of PV systems. Photovoltaic panels include one or more PV modules assembled as a pre-wired, field-installable unit. A photovoltaic array is the complete power-generating unit, consisting of any number of PV modules and panels. Irradiance and temperature are some of the factors that decide the efficiency of PV modules and hence can be used as parameters for fault detection in PV arrays. Irradiance is defined as the measure of the power density of sunlight received and is measured in watt per metre square. With the increasing solar irradiance both the open-circuit voltage and the short circuit current increase and hence the maximum power point varies. Temperature plays another major factor. As the temperature increases, the rate of photon generation increases thus reverse saturation current increases rapidly and this reduces the band gap. Hence this leads to marginal changes in current but major changes in voltage. Temperature acts like a negative factor affecting solar cell performance. Hence Temperature difference is used by us as main parameter for detection of faults, because defects and faults in PV modules and arrays almost always generate some temperature difference on the laminated semiconductor panel screen. 
<p align="center">
 <img src= "/assets/graphy.png">
 <p align="center">
 <i>Equivalent circuit of a PV cell</i><br> 
</p>

<p align="center">
 <img src= "/assets/graph.png">
 <p align="center">
 <i>Parameters of circuit</i><br> 
</p>
This temperature variation when taken as RGB 3-dim arrayed image matrix through thermal camera mounted on the UAV is used the feeding data for training of our YOLO model. </p> 


      
# MODEL AND ALGORITHM
<div>

### **<ins>YOLOv5</ins>**
<img src=
"/assets/yoLo.jpg"
         align="right" 
         width="150">
</div>

YOLO an acronym for 'You only look once', is an object detection algorithm that divides images into a grid system. Each cell in the grid is responsible for detecting objects within itself.
Today YOLO is one of the most famous object detection algorithms due to its speed and accuracy.

<p align="justify">
It uses a single neural network to process the entire picture, then separates it into parts and predicts bounding boxes and probabilities for each component. These bounding boxes are weighted by the expected probability. The method “just looks once” at the image in the sense that it makes predictions after only one forward propagation run through the neural network. It then delivers detected items after non-max suppression (which ensures that the object detection algorithm only identifies each object once).

Its architecture mainly consisted of three parts, namely:</p>

1. Backbone: Model Backbone is mostly used to extract key features from an input image. CSP(Cross Stage Partial Networks) are used as a backbone in YOLO v5 to extract rich in useful characteristics from an input image.
   
2. Neck: The Model Neck is mostly used to create feature pyramids. Feature pyramids aid models in generalizing successfully when it comes to object scaling. It aids in the identification of the same object in various sizes and scales.
   Feature pyramids are quite beneficial in assisting models to perform effectively on previously unseen data. Other models, such as FPN, BiFPN, and PANet, use various sorts of feature pyramid approaches.
PANet is used as a neck in YOLO v5 to get feature pyramids.

3. Head: The model Head is mostly responsible for the final detection step. It uses anchor boxes to construct final output vectors with class probabilities, objectness scores, and bounding boxes.

<p align="center">
 <img src= "/assets/yolov5.png">
 <p align="center">
 <i>YOLO ARCHITECTURE</i><br> 
</p>
   
-	YOLOv5 is specifically prefered above other YOLO version because YOLOv5 is about 88% smaller than YOLOv4 (27 MB vs 244 MB), It is about 180% faster than YOLOv4 (140 FPS vs 50 FPS) and is roughly as accurate as YOLOv4 on the same task (0.895 mAP vs 0.892 mAP).

- It was released shortly after the release of YOLOv4 by Glenn Jocher using the Pytorch framework on 18 May 2020
  
- Also YOLOv5 is also preferred over other detection models like R-CNN,Fast-RCNN and Faster-RCNN even though YOLO and Faster RCNN both share many similarities. They both uses a anchor box based network structure, both uses bounding both regression. Things that differs YOLO from Faster RCNN is that it makes classification and bounding box regression at the same time. Judging from the year they were published, it make sense that YOLO wanted a more elegant way to do regression and classification. YOLO however does have it’s drawback in object detection. YOLO has difficulty detecting objects that are small and close to each other due to only two anchor boxes in a grid predicting only one class of object. It doesn’t generalize well when objects in the image show rare aspects of ratio. Faster RCNN on the other hand, do detect small objects well since it has nine anchors in a single grid, however it fails to do real-time detection with its two step architecture.


<p align="center">
 <img src= "/assets/bus.jpg" align="center" width= "380">
 <p align="center">
 <i>Bounding boxes around detected objects</i><br> 
</p>

### **<ins>Dataset Preparations</ins>**

The basic data used for this project is **Photovoltaic thermal image dataset which was given to us by Robotics and Artificial Intelligence Department of Information Engineering Università Politecnica Delle Marche**. For its collection, a thermographic inspection of a ground-based PV system was carried out on a PV plant with a power of approximately 66 MW in Tombourke, South Africa. The thermographic acquisitions were made over seven working days, from 21 to 27 January 2019 with the sky predominantly clear and with maximum irradiation. This situation is optimal to enhance any abnormal behaviour(faults) of entire panels or any specific portion .

Dataset contains **3 folders** each containing **1009 images**. **1st folder** stores **pre-processed thermal images** taken by UAV copter, the **2nd folder** contains the **equivalent grayscale** images of the same thermal image taken through copter, while **3rd folder** contains **masked image** showing the separated single defected cells or contiguous sequence of faulty cells (string). Each folder contains images of size **512 X 640 pixels**. 
<div align="center">

<img src= "/assets/imgFC61.png" alt="Drone" align="left" width="230">

<img src= "/assets/mask61.png" alt="Drone" align="right" width="230">

<img src= "/assets/img61.png" alt="Drone" align="center" width="230">

</div>
<br><br>

<div align="center">

<img src= "/assets/imgFC59.png" alt="Drone" align="left" width="230">

<img src= "/assets/mask59.png" alt="Drone" align="right" width="230">

<img src= "/assets/img59.png" alt="Drone" align="center" width="230">

</div>
<br><br>
This data is modified in three different forms so as to obtain three different detection models for PV array, module and fault respectively. For YOLO to work we need bounding boxes around the object of consideration. YOLO need image along with the text file containing the coordinates of the bounding rectangle around the object of consideration.
We made bounding boxes around the faulty cells and strings using the masked images provided to us. But for PV array and single cell detection we made bounding boxes using app.roboflow.com .
<br>
<div align="center">
<img src= "/assets/roboflow2.jpeg" alt="Drone" align="center" width="430"></div>
<br>

For PV array we made on an average 4 bounding boxes per image for about 200 images. And on an average 30 bounding boxes per image for 72 images for single PV cell detection.


1. In depth specifications of the drone are [here](https://www.cyberbotics.com/doc/guide/mavic-2-pro?version=develop#mavic2pro-field-summary).
2. To know more about the sensors used in the drone, click [here](https://github.com/cyberbotics/webots/tree/released/docs/reference)


### **<ins>Predictions Made</ins>**

Predictions are reasonably acceptable. Detections do not have any false positives in them.

   #### **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <ins>PV array detection</ins>**
   
   - PV array detection has given the best results among all 3 detections during validations and testing. The model has almost perfect weights for PV array detection with an accuracy of. 
   
   <br>
    <div align="center">
    <img src= "/assets/array.jpeg" alt="Drone" align="center" width="430"></div>
   <br>
   
   #### **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <ins>Single PV modeule detection</ins>**
   
   - Single PV cell detection model predictions are also fairly accurate. However, sometimes it considers 2-3 PV modules as a single separated cell. And also sometimes leaves some boundary cells by not considering them as PV cells. Has accuracy of 

  <br>
    <div align="center">
    <img src= "/assets/single12.png" alt="Drone" align="center" width="430"></div>
   <br>

   #### **&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <ins>PV module with fault</ins>**
   
   - Faulty cell detections are also accurate for single defected cells and contiguous sequences of defective cells (string). The model has accuracy of 

  <br>
  <div align="center">
  <img src= "/assets/fault.jpeg" alt="Drone" align="center" width="430"></div>
  <br>
   
### **<ins>Model Setup Requirements</ins>**
| Requirements                 | Link(s)                                                          |
   |:-----------------:|:-----------------------------------------------------------------:|
   | <div align="center"><img src= "/assets/python.png" alt="Drone" align="center" width="70">| [python3](https://www.python.org/downloads/) |
   | <div align="center"><img src= "/assets/pytorch.png" alt="Drone" align="center">|[torch-python](https://pytorch.org/)|
   | <div align="center"><img src= "/assets/text.png" alt="Drone" align="center" width="130"> | [pakages](https://github.com/Pratyush-IITBHU/Automatic-Faults-Detection-of-Photovoltaic-Farms-using-Thermal-Images/blob/main/requirements.txt) |
   | <div align="center"><img src= "/assets/logo.png" alt="Drone" align="center" width="130" height="30"> | [Bounding box maker](https://app.roboflow.com/) |
   | <div align="center"><img src= "/assets/github.png" alt="Drone" align="center" width="100" > | [Exploratory Repository](https://github.com/Pratyush-IITBHU/Automatic-Faults-Detection-of-Photovoltaic-Farms-using-Thermal-Images) |


### **<ins>Initialisation</ins>**

   1. Gitclone the repsitory in your system.
      
   2. Open command prompt or terminal with your your specific python environment and type **pip install requirement.txt**.

   3. Now place the test thermal image to the false colour folder.
   

   4. Open the detect_teen.py file and run it with your python interpretor.
   5. Results will be displayed on a scree and will be stored inside the runs folder.

# Types of Faults
- ### Physical Faults:
  
  - **Soiling Fault:**
         <ul>
           <li>Detection of Fault due to dirt(because of various reasons) could be done easily as it will be warmer as compared to its surrounding.</li>
           <li>Could be detect if a local temperture variation is encountered while scanning.</li>
         </ul>

  <div align="center">
  <img src= "/assets/soil.png" alt="Drone" align="center" width="430"></div>
  <br>

  - **Diode Fault:**
         <ul>
           <li>High Resistance or other defects at specific parts of Diode screen of the panel(ex: bending etc)</li>
           <li>Can be detected if different spots of temperature variation are encountered within a single module</li>
         </ul>
  <br>    
  <div align="center">
  <img src= "/assets/diode.png" alt="Drone" align="center" width="430"></div>
  <br>

- ### Electrical Faults:
     
  <div align="center">
  <img src= "/assets/classi.png" alt="Drone" align="center" width="430"></div>
  <br>

  <br>    
  <div align="center">
  <img src= "/assets/trical_fault2.png" alt="Drone" align="center" width="400"></div>
  <br>
<br>
  <br>    
  <div align="center">
  <img src= "/assets/trical_fault.png" alt="Drone" align="center" width="430"></div>
  <br>

  <ul>
    <li>These above electrical faults moslty effect the whole PV array/string, hence the faults cooming in whole PV array can be considered as any of the electrical fault </li>
    <li>These errors can be confirmend from the parameters we can obtain throgh the inverter like string currents,Rate of change of string current, Dimensionally reducted data(PCA) of various parameters etc.
    </li>
    <li>Hence our further future works will based on the electrical fault detection using inverter parameters and ML models classifiers like SVM, k-nearest neigbour, random forest etc. And model with highest accuracy will be finalised.</li>
  </ul>


# Autonomous Drone Navigation

- For making UAV quadcopter fully autonomous and capable of taking images on its own, we have simulated scripts in ROS(noetic and ubuntu 20.0) and Gazebo(version-10).
   
- Drone URDF and plugins were made to test drone on simulation.

<br>    
  <div align="center">
  <img src= "/assets/droni1.png" alt="Drone" align="center" width="400"></div>
  <br>

- Navigation packages from ROS Navigation stack are added to Drone controller.Also Fine tuning mapping and navigating parameters is done for best results.
  
- Obstacle avoidance and path planning packages are also added to drone.

<div align="center">

<img src= "/assets/slam1.jpg" alt="Drone" align="left" width="200">

<img src= "/assets/slam2.jpg" alt="Drone" align="right" width="200">

<img src= "/assets/slam3.jpg" alt="Drone" align="center" width="200">

</div>
<br>

- However due to lack of thermal camera plugin in gazebo, complete testing on simulation of drone was done.

  <div align="center">
  <img src= "/assets/droni2.png" alt="Drone" align="center" width="400"></div>


# Conclusion

In this paper, YOLOv5, a novel deep learning model is used  for detecting faults in large-scale PV farms and plants. To achieve great results, improved Photovoltaic thermal image dataset was used along with proper hypertuned model. Proper detections are shown while code runs and results are properly saved. 

Proposed model(in future) is to be deployed in jetson which will be used as a companion computer in our UAV Copter. Copter design is almost complete in simuation(except thermal camera). 

Demo video can be seen from [HERE](https://dare2compete.com/o/b61Yrn2?lb=JqIDm9e)
         

# References

- [Fault detection, classification and protection in solar photovoltaic arrays](https://repository.library.northeastern.edu/files/neu:rx917d168/fulltext.pdf)

- [HDJSE_6960328 1..11 (hindawi.com)](https://downloads.hindawi.com/journals/js/2020/6960328.pdf)

- [Classification and Detection of Faults in Grid Connected Photovoltaic System (ijser.org)](https://www.ijser.org/researchpaper/Classification-and-Detection-of-Faults-in-Grid-Connected-Photovoltaic-System.pdf)

- [8712960.pdf (hindawi.com)](https://downloads.hindawi.com/journals/jece/2016/8712960.pdf)

- [2016-GOALI-Rep-1-Fault-Detection-using-Machine-Learning-in-PV-Arrays.pdf (asu.edu)](https://sensip.engineering.asu.edu/wp-content/uploads/2015/09/2016-GOALI-Rep-1-Fault-Detection-using-Machine-Learning-in-PV-Arrays.pdf)

- [energies-13-06496.pdf](https://www.google.com/url?q=https://www.mdpi.com/1996-1073/13/24/6496/pdf&sa=U&ved=2ahUKEwjml8Chz873AhXdSWwGHYVWB80QFnoECAsQAg&usg=AOvVaw2-zFawRuKSkcTZ-bZzIkoD)

- [Li, X.; Yang, Q.; Lou, Z.; Yan, W. Deep Learning Based Module Defect Analysis for Large-Scale Photovoltaic Farms. IEEE Trans. Energy Convers. 2018, 34, 520–529.](http://dx.doi.org/10.1109/TEC.2018.2873358)

- [Tsanakas, J.A.; Ha, L.; Buerhop, C. Faults and infrared thermographic diagnosis in operating c-Si photovoltaic modules: A review of research and future challenges.](http://dx.doi.org/10.1016/j.rser.2016.04.079)

- [Deep Learning Approach for Automated Fault Detection on Solar Modules Using Image Composites](https://ieeexplore.ieee.org/document/9518540)
