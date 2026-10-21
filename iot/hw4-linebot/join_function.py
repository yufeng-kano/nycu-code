# Paste into the IoTtalk GUI as the function of the join that feeds Msg-O_315834011.
# args follow the join's input order: Temperature_315834011, Humidity_315834011, Msg-I_315834011.
# Returns [question, reply] for Msg-O_315834011's two dimensions.
def run(*args):
    # Samples may arrive as one-element lists, depending on the server
    temperature, humidity, msg = [a[0] if isinstance(a, list) else a for a in args[:3]]
    if msg is None:
        return None
    text = str(msg).lower()
    if 'temp' in text or '溫度' in text:
        if temperature is None:
            return [msg, 'Temperature is not available yet.']
        return [msg, 'Current temperature: {:.1f} °C'.format(float(temperature))]
    if 'humid' in text or '濕度' in text or '溼度' in text:
        if humidity is None:
            return [msg, 'Humidity is not available yet.']
        return [msg, 'Current humidity: {:.1f} %'.format(float(humidity))]
    return [msg, 'Only temperature and humidity inquiries are supported.']
