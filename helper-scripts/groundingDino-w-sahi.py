import cv2
import supervision as sv

from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology

base_model = GroundingDINO(ontology=CaptionOntology({"cars": "private_car","motorcycle": "motorcycle","rickshaw": "rickshaw","rickshaw van": "rickshaw_van","truck": "truck","small trucks": "pickup_truck","bicycle":"cycle", "micro bus":"micro_bus", "bus": "bus"}))


base_model.label(
    "D:\\thesisVideo\\bns\\sahi",
    ".png",
    output_folder="D:\\thesisVideo\\bns\\sahi\\dataset",
    sahi=True,
)

# detections = base_model.sahi_predict("D:\\thesisVideo\\bns\\frames\\bns_243.png")
#
# classes = ["private_car", "motorcycle", "bus", "rickshaw", "truck", "pickup_truck"]
#
# box_annotator = sv.BoxAnnotator()
# label_annotator = sv.LabelAnnotator()
#
# labels = [
#     f"{classes[class_id]} {confidence:0.2f}"
#     for _, _, confidence, class_id, _, _
#     in detections
# ]
#
# image = cv2.imread("D:\\thesisVideo\\bns\\frames\\bns_243.png")
#
# annotated_frame = box_annotator.annotate(
#     scene=image.copy(),
#     detections=detections
# )
# annotated_frame = label_annotator.annotate(
#     scene=annotated_frame,
#     detections=detections,
#     labels=labels
# )
#
# sv.plot_image(image=annotated_frame, size=(16, 16))
