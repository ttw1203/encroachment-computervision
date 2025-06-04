from roboflow import Roboflow

rf = Roboflow(api_key="kbpgNiBrAULSrAW4nM8q")
workspace = rf.workspace("thesis-i7fy0")

workspace.deploy_model(
  model_type="yolov11",
  model_path="D:\\train_folder\\weights",
  project_ids=["bns-s7lvi"],
  model_name="customBrac",
  filename="D:\\train_folder\\weights\\best.pt"

)