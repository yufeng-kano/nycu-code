
import libraries.LineBot_315834011_library.safuncs as safuncs

ServerURL = 'https://class.iottalk.tw' #For example: 'https://DomainName'
MQTT_broker = 'iot.iottalk.tw' # MQTT Broker address, for example: 'DomainName' or None = no MQTT support
MQTT_port = 1883
MQTT_encryption = False
MQTT_User = 'iottalk'
MQTT_PW = 'iottalk2023'

device_model = 'LineBot_315834011'
IDF_list = []
ODF_list = ['Msg-O_315834011']
device_id = '13846234-efc0-44c6-96a8-89b0038e0a41'
device_name = 'LineBot_315834011'
exec_interval = 1  # IDF/ODF interval





def Msg_O_315834011(data):
    safuncs.Msg_O_315834011(data)


def on_register(r):
    print(f'Device name: {r["d_name"]}')    
    '''
    #You can write some SA routine code here, for example: 
    import time, DAI
    while True:
        DAI.push('Dummy_Sensor', [100, 200])  
        time.sleep(exec_interval)    
    '''


