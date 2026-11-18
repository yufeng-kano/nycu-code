
import libraries.M2_315834011_library.safuncs as safuncs

ServerURL = 'https://class.iottalk.tw' #For example: 'https://DomainName'
MQTT_broker = 'iot.iottalk.tw' # MQTT Broker address, for example: 'DomainName' or None = no MQTT support
MQTT_port = 1883
MQTT_encryption = False
MQTT_User = 'iottalk'
MQTT_PW = 'iottalk2023'

device_model = 'M2_315834011'
IDF_list = ['CO2_315834011', 'Humidity_315834011', 'Temperature_315834011']
ODF_list = []
device_id = '379c24e3-4233-40f0-8a6b-2cb32d097d38'
device_name = 'M2_315834011'
exec_interval = 1  # IDF/ODF interval



def CO2_315834011():
    return safuncs.CO2_315834011()

def Humidity_315834011():
    return safuncs.Humidity_315834011()

def Temperature_315834011():
    return safuncs.Temperature_315834011()




def on_register(r):
    print(f'Device name: {r["d_name"]}')    
    '''
    #You can write some SA routine code here, for example: 
    import time, DAI
    while True:
        DAI.push('Dummy_Sensor', [100, 200])  
        time.sleep(exec_interval)    
    '''


