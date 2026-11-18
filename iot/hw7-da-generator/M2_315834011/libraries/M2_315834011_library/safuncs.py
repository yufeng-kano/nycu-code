


# sensors.py (from M2_library.zip) must sit next to DAI.py in the DA folder
import sensors

co2 = sensors.sensorCO2_API()
co2.start()
tp_and_rh = sensors.sensorSHT31_API()  # (temperature, humidity)
tp_and_rh.start()


def CO2_315834011():
#    
    # None until the first farm reading arrives; DAI skips None


    return co2.getData()



# sensors.py (from M2_library.zip) must sit next to DAI.py in the DA folder
import sensors

co2 = sensors.sensorCO2_API()
co2.start()
tp_and_rh = sensors.sensorSHT31_API()  # (temperature, humidity)
tp_and_rh.start()


def Humidity_315834011():
#    
    t = tp_and_rh.getData()


    return t[1] if t else None



# sensors.py (from M2_library.zip) must sit next to DAI.py in the DA folder
import sensors

co2 = sensors.sensorCO2_API()
co2.start()
tp_and_rh = sensors.sensorSHT31_API()  # (temperature, humidity)
tp_and_rh.start()


def Temperature_315834011():
#    
    t = tp_and_rh.getData()


    return t[0] if t else None

