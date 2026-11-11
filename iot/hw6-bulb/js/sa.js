$(function () {
    var iottalk_url = 'https://class.iottalk.tw';
    var mqtt_url = 'wss://iot.iottalk.tw:8866/mqtts';
    var mqtt_user = 'iottalk';
    var mqtt_password = 'iottalk2023';
    var exec_interval = 500;

    // Bulb state: colour is 0-255 per channel, luminance is 0-100 %.
    var r = 255;
    var g = 255;
    var b = 0;
    var lum = 100;

    function clamp (value, max) {
        value = Number(value);
        if (isNaN(value)) return 0;
        return Math.min(Math.max(value, 0), max);
    }

    function draw () {
        var rr = Math.floor((r * lum) / 100);
        var gg = Math.floor((g * lum) / 100);
        var bb = Math.floor((b * lum) / 100);
        $('.bulb-top, .bulb-middle-1, .bulb-middle-2, .bulb-middle-3, .bulb-bottom, .night').css(
            {'background': 'rgb(' + rr + ', ' + gg + ', ' + bb + ')'}
        );
    }

    var profile = {
        'dm_name': 'Bulb',
        'idf_list': [],
        'odf_list': [Color_O, Luminance_O],   // dai.js maps the trailing _O to -O
        'd_name': undefined,
    };

    function Color_O (data) {
        r = clamp(data[0], 255);
        g = clamp(data[1], 255);
        b = clamp(data[2], 255);
        draw();
    }

    function Luminance_O (data) {
        lum = clamp(data[0], 100);
        draw();
    }

/*******************************************************************/
    function sa_init () {
        console.log(profile.d_name);
        $('.Device_name')[0].innerText = profile.d_name;
    }
    var sa = {
        'sa_init': sa_init,
        'iottalk_url': iottalk_url,
        'mqtt_url': mqtt_url,
        'mqtt_user': mqtt_user,
        'mqtt_password': mqtt_password,
        'exec_interval': exec_interval,
    };
    draw();
    dai(profile, sa);
});
