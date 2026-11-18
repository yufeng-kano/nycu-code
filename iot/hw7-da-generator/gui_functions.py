# Functions pasted into the IoTtalk GUI (project 315834011_HW7, Join 2).
# They run on the IoTtalk server, not locally; this file is a copy for reference.


# IDF function "315834011_hw7_label", selected for CO2, Humidity and Temperature.
# The server defines idf_df_name_ only inside IDF functions: it is the name of the
# IDF the function is attached to, so one function labels all three readings.
def run(*args):
    name = idf_df_name_.replace('_315834011', '')
    unit = {'Temperature': ' °C', 'Humidity': ' %', 'CO2': ' ppm'}.get(name, '')
    return [idf_df_name_, '{}: {:.1f}{}'.format(name, float(args[0]), unit)]


# Join function "315834011_hw7".
# Each argument is [idf_df_name_, text] from the IDF function above.
# Returns [source names, one message with every reading] for Msg-O's two dimensions.
def run(*args):
    pairs = [a for a in args if a]
    names = ', '.join(name for name, _ in pairs)
    text = '\n'.join(reading for _, reading in pairs)
    return [names, 'M2 weather station\n' + text]
