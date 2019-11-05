Exercising is moving between states. So we need following:
1] Pose detection model that can run at real time on cheap and easily available hardware
Hardware:
    1] JetsonNano, RaspberryPi, Mobile, EdgeTPU, Laptop
    2] Cost, availability, practicality
    3] For this demo, laptop + edgetpu
   
Handle variability of speed:
    1] Pick image only when processing of one is complete
    2] Run another thread with camera
    Camera.py
   
Training your model:
1] Train a model for segmentation which works at 30fps on CPU and is less than 4MB. 
Why ? Small models can be updated over network.
2] Tricks
    a] Pruning
    b] Quantization:fp16, int8
    c] TFLite
 
Models lose accuracy when we make them smaller and faster. So now we have 2 models, one accurate and another fast. 
We combine them to improve accuracy:
1] By training a kalman filter like structure. Can be trained online or offline
2] By running one accurate and one fast model in parallel
    a] Running models in parallel on different devices : CPU + GPU + TPU
    b] Feed accurate models input to kalman filter for better state updates
3] Measuring accuracy
    a] Two models running in parallel lets us compute runtime accuracy
        
2] Break an exercise into stages, and when a person goes through all stages of an exercise, the exercise finishes
    a] Match pose of person with target pose
    b] No standard metric to measure distance between two poses
    c] Three approaches:
        1.Train a classifier to classify poses, either use already available datasets
        2.Prepare own dataset by scraping google with yoga pose names
        3.Use videos to train a siamese network, frames which are apart in time in the video, should be less similar
        4.Directly use existing similarity metrics between pose vectors

3] Feedback
    a] What can we show the user ? 
        1. language feedback does not work [Experiment to try: make someone get in a pose by describing it, its difficult to describe "how much"]
        2. video is very difficult to follow
        
