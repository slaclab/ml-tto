import numpy as np
from time import sleep
import epics

def measure_background(screen, shutter_pv):
    epics.caput(shutter_pv,0) 
    sleep(1)
    
    background_images = []
    for i in range(20):
        background_images += [screen.image]
        sleep(0.2)
    
    background_image = np.mean(background_images, axis=0)
    
    epics.caput(shutter_pv,1) 
    sleep(2)

    return background_image