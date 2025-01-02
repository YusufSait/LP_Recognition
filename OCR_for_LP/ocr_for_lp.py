import pytesseract
import cv2
import matplotlib.pyplot as plt
import os
import re


TARGET_WIDTH_PX = 200

def pre_process_img(src_img):# Get original dimensions
	img_gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
	
	# Resize image
	_height, _width = img_gray.shape
	_aspect_ratio = _height / _width
	_target_height = int(TARGET_WIDTH_PX * _aspect_ratio)
	img_gray = cv2.resize(img_gray, (TARGET_WIDTH_PX, _target_height), interpolation = cv2.INTER_CUBIC)
	
	# img_gray = cv2.Laplacian(img_gray, cv2.CV_8UC1)
	img_gray = cv2.equalizeHist(img_gray)
	img_gray = cv2.GaussianBlur(img_gray, (1, 1), cv2.BORDER_DEFAULT)
	
	return img_gray

def extract_bbox_region(img, yolo_data):
    img_height, img_width, _ = img.shape

    _, x_center, y_center, width, height = yolo_data

    # Convert YOLO coordinates to pixel coordinates
    x_center = int(x_center * img_width)
    y_center = int(y_center * img_height)
    w = int(width * img_width)
    h = int(height * img_height)

    # Calculate top-left and bottom-right corner coordinates
    x1 = max(0, int(x_center - w / 2))
    y1 = max(0, int(y_center - h / 2))
    x2 = min(img_width, int(x_center + w / 2))
    y2 = min(img_height, int(y_center + h / 2))

    cropped_img = img[y1:y2, x1:x2]  # Extract the roi

    return cropped_img

def get_lp_paths(base_dir = "../AI_assigment_datasets/"):
	subfolders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
	subfolders = sorted(subfolders)
	file_pairs = []

	for subfolder in subfolders:
		image_dir = os.path.join(base_dir, subfolder, "image")
		annotation_dir = os.path.join(base_dir, subfolder, "annotation")

		# Get all filenames in both directories without extensions
		image_files = {os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith(".jpg")}
		annotation_files = {os.path.splitext(f)[0] for f in os.listdir(annotation_dir)}

		# Find matching file pairs
		common_files = image_files.intersection(annotation_files)
		for name in common_files:
			file_pairs.append( (os.path.join(image_dir, name + ".jpg"),
				os.path.join(annotation_dir, name + ".txt"))
			)
	return file_pairs

def check_lp_country(lp_text):
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
	
	for country_name, country_regex in patterns.items():
		if re.match(country_regex, lp_text):
			return country_name
	
	return "No country found!"

lp_file_pairs = get_lp_paths()

countries = ["Brazil", "Estonia", "Finland", "Kazakhstan", "Lithuania", "Serbia",
"UAE"]

for img_path, ann_path in lp_file_pairs:
	input_img = cv2.imread(img_path)
	yolo_data_list = []
	current_country = ""
	for country in countries:
		if country.lower() in ann_path.lower():
			current_country = country
	with open(ann_path, "r") as f:
		for line in f:
			row = list(map(float, line.split()))
			yolo_data_list.append(row)
	for yolo_data in yolo_data_list:
		lp_img = extract_bbox_region(input_img, yolo_data)
		lp_img = pre_process_img(lp_img)
		predicted_result = pytesseract.image_to_string(lp_img, config='-c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyz --psm 6', lang='eng')
		filter_predicted_result = "".join(predicted_result.split()).replace(":", "").replace("-", "")
	
	plt.title("Predicted: " + filter_predicted_result +
	 "\nPredicted country: " + check_lp_country(filter_predicted_result) +
	 "\n Actual country: " + current_country)
	plt.imshow(lp_img)
	plt.show()
