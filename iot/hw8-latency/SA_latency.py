import csv
import os
from time import perf_counter

PROTOCOL = os.environ.get('PROTOCOL', 'mqtt')  # 'mqtt' or 'http'
SAMPLES = int(os.environ.get('SAMPLES', '120'))
OUTPUT = f'latency_{PROTOCOL}.csv'

ServerURL = 'https://class.iottalk.tw'
MQTT_broker = 'class.iottalk.tw' if PROTOCOL == 'mqtt' else None  # None = HTTP
MQTT_port = 1883
MQTT_encryption = False
MQTT_User = 'iottalk'
MQTT_PW = 'iottalk2023'

device_model = 'Dummy_Device'
IDF_list = ['Dummy_Sensor']
ODF_list = ['Dummy_Control']
device_id = '315834011_HW8'
device_name = None
exec_interval = 1  # IDF/ODF interval

sent = set()  # t1 values sent by this process; filters stale data from earlier runs
latencies = []
if os.path.exists(OUTPUT):
    os.remove(OUTPUT)  # start a fresh run before any sample arrives


def Dummy_Sensor():
    t1 = perf_counter()
    sent.add(t1)
    return t1


def Dummy_Control(data: list):
    t2 = perf_counter()
    t1 = data[0]
    if t1 not in sent:
        print('skip stale sample', data)
        return
    sent.discard(t1)
    latency = t2 - t1
    latencies.append(latency)
    print(f'[{PROTOCOL}] #{len(latencies)} latency = {latency * 1000:.1f} ms')
    with open(OUTPUT, 'a', newline='') as f:
        csv.writer(f).writerow([len(latencies), t1, t2, latency])
    if len(latencies) >= SAMPLES:
        print(f'Collected {SAMPLES} samples -> {OUTPUT}')
        os._exit(0)


def on_register(r):
    print(f'Device name: {r["d_name"]} ({PROTOCOL})')
