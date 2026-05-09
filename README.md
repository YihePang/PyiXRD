
### 1.There are two methods to obtain the crystal XRD patterns dataset:
&emsp;&emsp;(1) Obtain the data by contacting the author.  
&emsp;&emsp;(2) Directly run the script patterns2xrd.py. A complete XRD dataset will be generated in the XRD_patterns folder. **Note:** The data acquired by this script will change along with updates to the Materials Project database.
### 2.Follow the procedure below to invoke the model for experiments:
&emsp;&emsp;(1) Ensure the complete dataset is stored in the XRD_patterns folder.  
&emsp;&emsp;(2) (Optional) When using test_mcnn.py for model testing, you may contact the authors to obtain the model .pth weight files, or download the two required model .pth weights via the links below and place them in the model_pth folder:  
&emsp;&emsp;https://huggingface.co/Yi-he/pyixrd/resolve/main/crystal_system.pth  
&emsp;&emsp;https://huggingface.co/Yi-he/pyixrd/resolve/main/space_group.pth  
&emsp;&emsp;(3) Create a new Python virtual environment using symmetry_conda_env.yml and install the required third-party Python libraries with the following commands:  
&emsp;&emsp;conda env create -f symmetry_conda_env.yml  
&emsp;&emsp;conda activate symmetry_conda_env  
&emsp;&emsp;(4) Run the mcnn.py script within the symmetry_conda_env virtual environment.
### 3.Notes:
&emsp;&emsp;(1) All .py files including patterns2xrd.py, mcnn.py, Loss.py, loaddata.py, evaluation_index.py and test_mcnn.py, as well as the folders XRD_patterns, model_pth, split_data and result_files, must be placed under the same file path.  
&emsp;&emsp;(2) The provided test_mcnn.py script is used to test the optimal model weight .pth files in the model_pth folder. This script is only applicable to the XRD test set. Use mcnn.py for the complete model training, validation and testing.  
&emsp;&emsp;(3) If the program reports an error indicating missing dependent libraries during runtime, please manually install the required libraries.
