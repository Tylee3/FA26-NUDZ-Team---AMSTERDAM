"""Aligns the SIGMA in vivo template to the Waxholm template once (atlas to atlas), so a rat's
existing Waxholm alignment can carry SIGMA's labels onto the rat.

Both are complete, clean brain templates, so a full affine fit (which can stretch, unlike
the rigid fit used on the thick-sliced rat scans) is well-posed here: it absorbs the size
difference between the two rat strains (Sprague Dawley vs Wistar). SIGMA's own published
Waxholm transform targets its ex vivo space, not the in vivo template we use, so it is
not used here.

The fit is checked on the atlases alone (no rats): both atlases use the same label ids,
so after alignment their whole-brain masks and hypothalamus (label 48) should overlap well.

Writes sigma/waxholm_to_sigma.tfm (maps a Waxholm-space point to a SIGMA-space point).
"""

import numpy as np
import SimpleITK as sitk

from atlas_registration_pipeline import ATLAS_LABELS as WAX_LABELS
from atlas_registration_pipeline import ATLAS_MRI as WAX_TEMPLATE
from sigma_registration import SIGMA_ATLAS, SIGMA_DIR, SIGMA_TEMPLATE

TRANSFORM_PATH = SIGMA_DIR / "waxholm_to_sigma.tfm"
SIGMA_MASK = SIGMA_DIR / "SIGMA_InVivo_Anatomical_Brain_mask.nii.gz"


def _register(fixed, moving, fixed_mask, moving_mask, initial):
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(50)
    reg.SetMetricFixedMask(fixed_mask)
    reg.SetMetricMovingMask(moving_mask)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.2, seed=1)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0, minStep=1e-4,
                                                 numberOfIterations=300, relaxationFactor=0.5)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(initial, inPlace=True)
    reg.Execute(fixed, moving)
    return initial


def align():
    wax = sitk.Cast(sitk.ReadImage(str(WAX_TEMPLATE)), sitk.sitkFloat32)
    sig = sitk.Cast(sitk.ReadImage(str(SIGMA_TEMPLATE)), sitk.sitkFloat32)
    wax_labels = sitk.ReadImage(str(WAX_LABELS))
    wax_mask = sitk.Cast(sitk.Resample(wax_labels, wax, sitk.Transform(), sitk.sitkNearestNeighbor, 0) > 0, sitk.sitkUInt8)
    sig_mask = sitk.Cast(sitk.ReadImage(str(SIGMA_MASK)) > 0, sitk.sitkUInt8)

    init = sitk.CenteredTransformInitializer(wax_mask, sig_mask, sitk.Similarity3DTransform(),
                                             sitk.CenteredTransformInitializerFilter.MOMENTS)
    sim = _register(wax, sig, wax_mask, sig_mask, sitk.Similarity3DTransform(init))
    affine = sitk.AffineTransform(3)
    affine.SetMatrix(sim.GetMatrix())
    affine.SetTranslation(sim.GetTranslation())
    affine.SetCenter(sim.GetCenter())
    final = _register(wax, sig, wax_mask, sig_mask, affine)
    sitk.WriteTransform(final, str(TRANSFORM_PATH))
    return final


def check(transform):
    """Dice between Waxholm's own masks and SIGMA's masks pulled into Waxholm space."""
    wax_labels = sitk.ReadImage(str(WAX_LABELS))
    sig_atlas = sitk.ReadImage(str(SIGMA_ATLAS))
    pulled = sitk.Resample(sig_atlas, wax_labels, transform, sitk.sitkNearestNeighbor, 0)
    w = sitk.GetArrayFromImage(wax_labels)
    s = np.rint(sitk.GetArrayFromImage(pulled)).astype(int)
    dice = lambda a, b: 2 * (a & b).sum() / (a.sum() + b.sum())
    shared = sorted(set(np.unique(w)) & set(np.unique(s)) - {0})
    per_label = [dice(w == k, s == k) for k in shared]
    print(f"whole brain Dice {dice(w > 0, s > 0):.3f} | hypothalamus (48) Dice {dice(w == 48, s == 48):.3f} | "
          f"median over {len(shared)} shared labels {np.median(per_label):.3f}")


if __name__ == "__main__":
    print("before alignment:", end=" ")
    check(sitk.Transform())
    t = align()
    m = np.array(t.GetParameters()[:9]).reshape(3, 3)
    print("affine scale factors (singular values):", np.round(np.linalg.svd(m, compute_uv=False), 3))
    print("after alignment: ", end=" ")
    check(t)
