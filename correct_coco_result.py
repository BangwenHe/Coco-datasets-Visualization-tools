import json


if __name__ == "__main__":
    coco_annotation_path = "D:/Dataset/edgePersonalDataset8IntCOCO/annotations/instances_val2017.json"

    coco_content = json.load(open(coco_annotation_path, "r"))
    coco_annotation = coco_content["annotations"]

    for anno in coco_annotation:
        pass
