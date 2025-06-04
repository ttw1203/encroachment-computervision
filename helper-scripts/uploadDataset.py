import roboflow

rf = roboflow.Roboflow(api_key="yHCY9O5npNAzasgmBvYZ")

# get a workspace
workspace = rf.workspace("thesis-i7fy0")

# Upload data set to a new/existing project
workspace.upload_dataset(
    "C:\\Users\\bfp14\\Downloads\\Compressed\\Vehicle_classification.v3i.yolov11", # This is your dataset path
    "customtraining-fs9si", # This will either create or get a dataset with the given ID
    num_workers=10,
    project_license="MIT",
    project_type="object-detection",
    batch_name=None,
    num_retries=0
)