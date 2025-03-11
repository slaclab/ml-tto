import datetime
from epics import caget_many
import numpy as np


def isotime():
    return (
        datetime.datetime.now()
        .replace(tzinfo=datetime.timezone.utc)
        .astimezone()
        .replace(microsecond=0)
        .isoformat()
    )


isotime()

run_dir = "./"


def numpy_save(path=run_dir + "Data/", spec="otr2"):
    pvname_list = [
        "SOLN:IN20:121:BACT",
        "QUAD:IN20:121:BACT",
        "QUAD:IN20:122:BACT",
        "QUAD:IN20:361:BACT",
        "QUAD:IN20:371:BACT",
        "QUAD:IN20:425:BACT",
        "QUAD:IN20:441:BACT",
        "QUAD:IN20:511:BACT",
        "QUAD:IN20:525:BACT",
        "SOLN:IN20:121:BCTRL",
        "QUAD:IN20:121:BCTRL",
        "QUAD:IN20:122:BCTRL",
        "QUAD:IN20:361:BCTRL",
        "QUAD:IN20:371:BCTRL",
        "QUAD:IN20:425:BCTRL",
        "QUAD:IN20:441:BCTRL",
        "QUAD:IN20:511:BCTRL",
        "QUAD:IN20:525:BCTRL",
    ]

    img_list_otr2 = [
        "OTRS:IN20:571:Image:ArrayData",
        "OTRS:IN20:571:XRMS",
        "OTRS:IN20:571:YRMS",
        "OTRS:IN20:571:Image:ArraySize1_RBV",
        "OTRS:IN20:571:Image:ArraySize0_RBV",
        "OTRS:IN20:571:X",
        "OTRS:IN20:571:Y",
        "OTRS:IN20:571:N_OF_BITS",
        "OTRS:IN20:571:RESOLUTION",
    ]

    img_list_otr3 = [
        "OTRS:IN20:621:IMAGE",
        "OTRS:IN20:621:XRMS",
        "OTRS:IN20:621:YRMS",
        "OTRS:IN20:621:ROI_XNP",
        "OTRS:IN20:621:ROI_YNP",
        "OTRS:IN20:621:X",
        "OTRS:IN20:621:Y",
        "OTRS:IN20:621:RESOLUTION",
    ]

    img_list_vcc = [
        "CAMR:IN20:186:IMAGE",
        "CAMR:IN20:186:N_OF_ROW",
        "CAMR:IN20:186:N_OF_COL",
        "CAMR:IN20:186:Y",
        "CAMR:IN20:186:X",
        "CAMR:IN20:186:YRMS",
        "CAMR:IN20:186:XRMS",
        "CAMR:IN20:186:RESOLUTION",
    ]

    if spec == "otr2":
        img_list = img_list_otr2 + img_list_vcc

    if spec == "otr3":
        img_list = img_list_otr3 + img_list_vcc

    ts = isotime()

    values = caget_many(pvname_list, timeout=10.0)
    imgs = caget_many(img_list, timeout=10.0)
    # save separately if have a long pv list of scalars to parse
    np.save(path + "values_" + ts + ".npy", dict(zip(pvname_list, values)))
    np.save(path + "imgs_" + ts + ".npy", dict(zip(img_list, imgs)))

    return dict(zip(pvname_list, values)), dict(zip(img_list, imgs))
