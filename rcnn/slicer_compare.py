"""Loads one rat's MRI with the manual trace, the atlas result and each R-CNN model's
prediction as separately colored outlines in 3D Slicer.

In Slicer: View > Python Console, then paste this line (with your own path to the repo):

    exec(open("/Users/<you>/Downloads/hypo_segments/rcnn/slicer_compare.py").read())

Change RAT to view a different test rat (1, 3, 4, 5, 6, 7, 8, 9, 10) and SHOW to limit
the view to the models you're assigned. Toggle any layer with its eye icon in the
Data module.
"""

import os

import slicer

RAT = 3
ROOT = os.path.expanduser("~/Downloads/hypo_segments")
SHOW = ["rcnn", "fast_rcnn", "faster_rcnn", "rfcn", "cascade_rcnn", "mask_rcnn"]

COLORS = {
    "manual (Me)": (0.0, 1.0, 0.0),
    "atlas": (1.0, 1.0, 1.0),
    "rcnn": (1.0, 0.55, 0.0),
    "fast_rcnn": (1.0, 0.0, 1.0),
    "faster_rcnn": (0.0, 1.0, 1.0),
    "rfcn": (0.6, 0.4, 1.0),
    "cascade_rcnn": (0.2, 0.4, 1.0),
    "mask_rcnn": (1.0, 0.1, 0.1),
}

sid = f"{RAT:03d}"
slicer.mrmlScene.Clear(0)
volume = slicer.util.loadVolume(os.path.join(ROOT, "T2 for segmentation", f"sub-{sid}_ses-1_acq-RARE_T2w.nii"))


def add_outline(path, name):
    if not path or not os.path.exists(path):
        print(f"  missing: {name} ({path})")
        return
    label = slicer.util.loadLabelVolume(path)
    seg = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", name)
    seg.SetReferenceImageGeometryParameterFromVolumeNode(volume)
    slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(label, seg)
    slicer.mrmlScene.RemoveNode(label)
    segmentation = seg.GetSegmentation()
    if segmentation.GetNumberOfSegments() == 0:
        print(f"  {name}: predicted nothing for rat {RAT}")
        return
    for i in range(segmentation.GetNumberOfSegments()):
        segment = segmentation.GetNthSegment(i)
        segment.SetName(name)
        segment.SetColor(*COLORS[name])
    seg.CreateDefaultDisplayNodes()
    display = seg.GetDisplayNode()
    display.SetVisibility2DFill(False)
    display.SetSliceIntersectionThickness(3)
    seg.CreateClosedSurfaceRepresentation()


manual_dir = os.path.join(ROOT, "Manual Segments", "Segmented nrrd", "Me")
manual = next((os.path.join(manual_dir, f) for f in os.listdir(manual_dir)
               if f.startswith(f"{RAT}_Segmentation")), None)
add_outline(manual, "manual (Me)")
add_outline(os.path.join(ROOT, "auto_seg_out", f"auto_seg_{sid}.nrrd"), "atlas")
for model in SHOW:
    add_outline(os.path.join(ROOT, "rcnn_out", model, f"pred_{sid}.nrrd"), model)

slicer.util.setSliceViewerLayers(background=volume, fit=True)
print(f"rat {RAT} loaded. Colors: " + ", ".join(f"{k}" for k in ["manual (Me)", "atlas", *SHOW]))
print("green = manual trace, white = atlas, then orange R-CNN, magenta Fast, cyan Faster, "
      "purple R-FCN, blue Cascade, red Mask R-CNN")
