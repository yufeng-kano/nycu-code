


# LineBot.py, config.py and userIdFile (LineBot-basic) must sit next to DAI.py
import time
import LineBot

# The join fires on every sensor update (about once a second), but the free
# LINE plan allows only 200 pushes a month, so send at most one per interval
PUSH_INTERVAL = 600  # seconds
last_push = 0


def Msg_O_315834011(data):
#    
    global last_push
    sources, message = data  # from join function 315834011_hw7
    print('Msg-O from', sources)
    if time.time() - last_push < PUSH_INTERVAL:
        return
    last_push = time.time()
    LineBot.pushLineMsg(message)




