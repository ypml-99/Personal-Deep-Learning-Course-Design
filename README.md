# Personal-Deep-Learning-Course-Design
Personal learning use
## Model operation steps

Here are the detailed steps on how to run our deep learning model 'best.pt':

### Environmental preparation

1. **Install Python**:
   Ensure that Python 3.8 is installed on your system. You can access it from the Python official website(https://www.python.org/downloads/)Download and install.

2. **Install dependencies**:
   Open a terminal or command prompt in the root directory of the project, and run the following command to install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

### Model operation

1. **Download pre trained model**:
   Make sure you have downloaded the '2022337621104Deep Learning.pt' file and placed it in the root directory of the project.

2. **Preparing to input data**:
   Place the images you want to predict in a folder, such as input_images.

3. **Run file**
   Modify the input file path of predict.py to ensure it matches your image path,then run.

### Output explanation

   The model will output prediction results, including bounding boxes and category labels for each detected object. The output will be saved in the 'runs' folder.
