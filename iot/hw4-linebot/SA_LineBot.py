import threading
import LineBot  # from LineBot-basic; its `import config` picks up ./config.py

ServerURL = 'https://class.iottalk.tw'
MQTT_broker = 'class.iottalk.tw'
MQTT_port = 1883
MQTT_encryption = False
MQTT_User = 'iottalk'
MQTT_PW = 'iottalk2023'

device_model = 'LineBot_315834011'
IDF_list = ['Msg-I_315834011']
ODF_list = ['Msg-O_315834011']
device_id = '315834011_LineBot'
device_name = None
exec_interval = 1  # IDF/ODF interval

# LineBot.init() blocks on Flask, while DAI blocks on MQTT, so Flask gets its own thread
threading.Thread(target=LineBot.init, kwargs={'port': 32768}, daemon=True).start()

# The join re-fires on every temperature/humidity update, so Msg-O_315834011 echoes the question
# and only the reply to the question still waiting for an answer goes back to LINE
pending_question = None

def Msg_I_315834011():
    global pending_question
    if not LineBot.msg_queue:
        return None
    pending_question = LineBot.msg_queue.pop(0)
    return pending_question

def Msg_O_315834011(data):
    global pending_question
    question, reply = data
    if pending_question is None or str(question) != pending_question:
        return
    pending_question = None
    LineBot.pushLineMsg(reply)

def on_register(r):
    print(f'Device name: {r["d_name"]}')
