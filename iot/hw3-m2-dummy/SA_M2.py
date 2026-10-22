import sensors

ServerURL = 'https://class.iottalk.tw'
MQTT_broker = 'class.iottalk.tw'
MQTT_port = 1883
MQTT_encryption = False
MQTT_User = 'iottalk'
MQTT_PW = 'iottalk2023'

device_model = 'M2_315834011'
IDF_list = ['CO2_315834011', 'Humidity_315834011', 'Temperature_315834011']
ODF_list = []
device_id = '315834011_M2'
device_name = None
exec_interval = 1  # IDF/ODF interval

# Real sensor values from the farm (see sensors.py)
co2 = sensors.sensorCO2_API()
co2.start()
tp_and_rh = sensors.sensorSHT31_API()
tp_and_rh.start()

def CO2_315834011():
    # Returns None (skip this round) until the first farm reading arrives
    return co2.getData()

def Humidity_315834011():
    t = tp_and_rh.getData()  # (temperature, humidity)
    return t[1] if t else None

def Temperature_315834011():
    t = tp_and_rh.getData()
    return t[0] if t else None

def on_register(r):
    print(f'Device name: {r["d_name"]}')
