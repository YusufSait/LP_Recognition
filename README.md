# <p align="center">ETE Tech - AI Assignment</p>
## <p align="center">Yusuf Sait ERDEM</p>

### License Plate Recognition
The YOLOv7 object detection model is trained by transfer-learning method on annotated license plate dataset where 50 images are split for training, 10 for validation, and 10 for testing.
* Pre-trained model test results:
	- TP, FP, FN, TN = 6, 4, 19, 0
	- precision = 6/25 = %24
* Transfer learning model test results:
	- TP, FP, FN, TN = 13, 5, 12, 0
	- precision = 13/25 = %52

The YOLOv7 code is derived from https://github.com/WongKinYiu/yolov7/ with minor changes. The pre-trained YOLOv7 model “LP_detect_yolov7_500img.pt” is downloaded from: https://github.com/mrzaizai2k/License-Plate-Recognition-YOLOv7-and-CNN

A license plate dataset in COCO format is generated. The dataset folder “AI_assigment_datasets” should be downloaded and placed next to “yolov7” project folder. All other necessary files are in this repository.

Command for training the pre-trained model with the “AI_assigment_datasets”:
```
$ python train.py --workers 7 --device 0 --batch-size 4 --epochs 50 --data data/LP_DS.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights 'LP_detect_yolov7_500img.pt' --name yolov7-LP --hyp data/hyp.scratch.p5.yaml
```
Command for training a plain model with “AI_assigment_datasets”:
```
$ python train.py --workers 8 --device 0 --batch-size 4 --epochs 10 --data data/LP_DS.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights '' --name yolov7-LP --hyp data/hyp.scratch.p5.yaml
```
Command for running the model on every images in a folder:
```
$ python detect.py --weights best.pt --conf 0.25 --img-size 640 --source your/images/folder/path
```

### Optical Character Recognition (OCR) for License Plates
Inside “OCR_for_LP” folder there is “ocr_for_lp.py” python code for extracting license code and origin country information from “AI_assigment_datasets”. The license plate ROI is extracted and analyzed to extract the required information. Following regex queries are utilized to determine the origin country:
```
patterns = {
		"Brazil": r"[a-zA-Z]{3}[a-zA-Z0-9]{4}",	# 3 letters - 4 digits or letters
		"Serbia": r"[a-zA-Z]{2}[0-9]{3,4}[a-zA-Z]{2}",  # 2 letters - 4-3 digits - 2 letters
		"Finland or Lithuania": r"[a-zA-Z]{3}[0-9]{3}",	 # 3 letters - 3 digits
		"Lithuania": r"[a-zA-Z]{3}[0-9]{3}",  # 2 letters - 3 digits
		"Estonia": r"[0-9]{2,3}[a-zA-Z]{3}",  # 2-3 digits - 3 letters
		"Estonia": r"[a-zA-Z]{2}[0-9]{4}",  # 2 letters 4 digits
		"Kazakhstan": r"[a-zA-Z][0-9]{3}[a-zA-Z]{3}",  # 1 letter - 3 digits  - 3 letters
		"Kazakhstan": r"[0-9]{3}[a-zA-Z]{2,3}[0-9]{2}",  # 3 digits - 3-2 letters - 2 digits
		"UAE": r"[0-9]{5,4}",  # 5-4 digits
		"UAE": r"[a-zA-Z][0-9]{5}",  # 1 letter - 5 digits
	}
```
Command to start the application: 
```
$ python ocr_for_lp.py
```
A screenshot from the application:
![LP_country-01](https://github.com/user-attachments/assets/c101732d-d757-4421-b72d-31f621804a99)

