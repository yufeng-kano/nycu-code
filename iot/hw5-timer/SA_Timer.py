from datetime import datetime as dt
import DAN

ServerURL = 'https://class.iottalk.tw'
MQTT_broker = 'class.iottalk.tw'
MQTT_port = 1883
MQTT_encryption = False
MQTT_User = 'iottalk'
MQTT_PW = 'iottalk2023'

device_model = 'Timer'
IDF_list = ['Time1', 'Time2', 'Time3']
ODF_list = []
device_id = '315834011_Timer'
device_name = None
exec_interval = 1  # IDF/ODF interval

def to_today(t):
    # Accept both '18:00' and '18:00:00'
    fmt = '%Y-%m-%d %H:%M:%S' if t.count(':') == 2 else '%Y-%m-%d %H:%M'
    return dt.strptime('{} {}'.format(dt.now().date(), t.strip()), fmt)

def timer(df_name):
    # The alias of the DF holds the setting, e.g. '18:00~18:05'
    alias = DAN.get_alias(df_name)
    if not alias or '~' not in alias[0]:
        return 0  # not bound yet, or the alias is still the DF name
    tlist = alias[0].split('~')
    try:
        begin, end = to_today(tlist[0]), to_today(tlist[1])
    except ValueError:
        print(f'{df_name}: cannot parse alias "{alias[0]}"')
        return 0
    now = dt.now()
    if begin <= end:
        return int(begin <= now < end)
    return int(now >= begin or now < end)  # window crosses midnight, e.g. 23:00~01:00

def Time1():
    return timer('Time1')

def Time2():
    return timer('Time2')

def Time3():
    return timer('Time3')

def on_register(r):
    print(f'Device name: {r["d_name"]}')
