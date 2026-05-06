
### 1.There are two methods to obtain the crystal XRD patterns dataset:
(1) Obtain the data by contacting the author.  
(2) Directly run the patterns2xrd.py script. A complete XRD patterns dataset will be generated in the directory where the script is located, and all patterns will be saved in the XRD_patterns folder.

### 2.Follow the procedure below to invoke the model for experiments:
(1) Ensure the complete dataset is stored in the XRD_patterns folder.  
(2) Contact the author to obtain the model .pth weight files, or download the two required model weight files via the links below and place them in the model_pth folder.  
https://huggingface.co/Yi-he/pyixrd/resolve/main/crystal_system.pth  
https://huggingface.co/Yi-he/pyixrd/resolve/main/space_group.pth  
(3) After installing all required Python third-party libraries for the project, run the mcnn.py script.

### 3.Note:
All .py files (patterns2xrd.py, mcnn.py, evaluation_index.py) and folders (XRD_patterns, model_pth, split_data) must be placed under the same directory path.
