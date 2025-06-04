import supervision as sv
import cv2
import os

from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology

base_model = GroundingDINO(ontology=CaptionOntology({"vehicles":"vehicles"}), box_threshold=0.3, text_threshold=0.25)

base_model.label("D:\\thesisVideo\\bns\\frames",extension=".png",output_folder="D:\\thesisVideo\\bns\\dataset")
# DATASET_NAME = "D:\\thesisVideo\\bns\\frames\\bns_051_dino.png"
# IMAGE_NAME = "D:\\thesisVideo\\bns\\frames\\bns_051.png"
#
# image = os.path.join(DATASET_NAME, IMAGE_NAME)
#
# predictions = base_model.predict(image)
#
# image = cv2.imread(image)
#
# annotator = sv.BoxAnnotator()
#
# annotated_image = annotator.annotate(scene=image, detections=predictions)
#
# sv.plot_image(annotated_image)