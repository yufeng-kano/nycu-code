import re, time, json, threading, requests, traceback
from datetime import datetime
import paho.mqtt.client as mqtt

MQTT_broker = 'farm.iottalk.tw' 
MQTT_port = 1883 
MQTT_User = 'iottalk'
MQTT_PW = 'iottalk2023'
MQTT_encryption = False

d_id = '08002700A202'

def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code.is_failure:
        print(f"Failed to connect: {reason_code}. loop_forever() will retry connection")
    else:
        print('Connected: {}'.format(reason_code))
        topic_list=[(d_id+'//AtPressure',0), (d_id+'//Humidity1',0), (d_id+'//Temperature1',0), (d_id+'//CO2',0), (d_id+'//Humidity2',0), (d_id+'//Temperature2',0), (d_id+'//UV1',0), (d_id+'//Luminance',0), (d_id+'//Infrared',0), (d_id+'//Moisture1',0), (d_id+'//SoilEC-I',0), (d_id+'//SoilTemp-I',0), (d_id+'//PH1',0), (d_id+'//WindSpeed',0), (d_id+'//WindDir',0), (d_id+'//RainMeter',0)]
        client.subscribe(topic_list)

def on_subscribe(client, userdata, mid, reason_code_list, properties):
    # Since we subscribed only for a single channel, reason_code_list contains
    # a single entry
    if reason_code_list[0].is_failure:
        print(f"Broker rejected you subscription: {reason_code_list[0]}")
    else:
        #print(f"Broker granted the following QoS: {reason_code_list[0].value}") 
        pass    
        
def on_disconnect(client, userdata, disconnect_flags, reason_code, properties):
    # paho-mqtt 2.x callback signature; loop_forever() reconnects automatically
    print('MQTT Disconnected. Re-connect...')

sensors={}
def on_message(client, userdata, msg):
    global sensors
    samples = json.loads(msg.payload)
    DF_name = msg.topic.split('//')[1]
    DF_data = samples['samples'][0][1][0]
    sensors[DF_name] = DF_data
    #print('{}: {}\n{}'.format(DF_name, DF_data, sensors))
    

def MQTT_config(client):
    client.username_pw_set(MQTT_User, MQTT_PW)
    client.on_connect = on_connect
    client.on_subscribe = on_subscribe
    client.on_message = on_message
    client.on_disconnect = on_disconnect
    if MQTT_encryption: client.tls_set()
    client.connect(MQTT_broker, MQTT_port, keepalive=60)

mqttc = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
MQTT_config(mqttc)
#mqttc.loop_start()

mrtt_p = threading.Thread(target=mqttc.loop_forever)
mrtt_p.daemon = True 
mrtt_p.start()


class sensorCO2_API:
    def __init__(self):
        pass

    state = 'SUSPEND'
    def start(self):
        self.state = 'RESUME'

    def getData(self):
        if self.state == 'RESUME': return sensors.get('CO2')
        else: return None

class sensorPH_API:
    def __init__(self):
        pass

    state = 'SUSPEND'
    def start(self):
        self.state = 'RESUME'

    def getData(self):
        if self.state == 'RESUME': return sensors.get('PH1')
        else: return None

class sensorPS_API:
    def __init__(self):
        pass

    state = 'SUSPEND'
    def start(self):
        self.state = 'RESUME'

    def getData(self):
        if self.state == 'RESUME':
            A = sensors.get('AtPressure')
            B = sensors.get('Temperature1')
            C = sensors.get('Humidity1')
            if None in [A, B, C]: return None
            return A, B, C
        else: return None

class sensorEC_API:
    def __init__(self):
        pass

    state = 'SUSPEND'
    def start(self):
        self.state = 'RESUME'

    def getData(self):
        if self.state == 'RESUME':
            A = sensors.get('Moisture1')
            B = sensors.get('SoilTemp-I')
            C =  sensors.get('SoilEC-I')
            if None in [A, B, C]: return None
            return A, B, C
        else: return None

class sensorSHT31_API:
    def __init__(self):
        pass

    state = 'SUSPEND'
    def start(self):
        self.state = 'RESUME'

    def getData(self):
        if self.state == 'RESUME':
            A = sensors.get('Temperature2')
            B = sensors.get('Humidity2')
            if None in [A, B]: return None
            return A, B
        else: return None

class sensorUV_API:
    def __init__(self):
        pass

    state = 'SUSPEND'
    def start(self):
        self.state = 'RESUME'

    def getData(self):
        if self.state == 'RESUME':
            A = sensors.get('UV1')
            B = sensors.get('Luminance')
            C =  sensors.get('Infrared')
            if None in [A, B, C]: return None
            return A, B, C
        else: return None


if __name__ == '__main__':
    pass
