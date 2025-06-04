from roboflow import Roboflow

rf = Roboflow(api_key="kbpgNiBrAULSrAW4nM8q")
project = rf.workspace().project("bns-vewvs")

#can specify weights_filename, default is "weights/best.pt"
version = project.version("1")

#example1 - directory path is "training1/model1.pt" for yolov8 model
version.deploy("rfdetr-base", "C:\\Users\\bfp14\\Downloads", "C:\\Users\\bfp14\\Downloads\\weights.pt")