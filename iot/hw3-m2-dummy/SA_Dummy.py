ServerURL = 'https://class.iottalk.tw'
MQTT_broker = 'class.iottalk.tw'
MQTT_port = 1883
MQTT_encryption = False
MQTT_User = 'iottalk'
MQTT_PW = 'iottalk2023'

device_model = 'Dummy_Device'
IDF_list = []
ODF_list = ['Dummy_Control']
device_id = '315834011_Dummy'
device_name = None
exec_interval = 1  # IDF/ODF interval

def Dummy_Control(data: list):
    # Values from M2_315834011 arrive here through the join in the IoTtalk GUI
    print(data)

def on_register(r):
    print(f'Device name: {r["d_name"]}')
