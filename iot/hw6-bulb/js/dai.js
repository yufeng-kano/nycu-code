const dai = function (profile, sa) {
    var odf_func = {};
    var idf_func = {};
    var mqtturl = sa.mqtt_url;
    var user = sa.mqtt_user;
    var password = sa.mqtt_password;
    var mqtt_client = undefined;
    var mac_addr = (function () {
        function s () {
            return Math.floor((1 + Math.random()) * 0x10000)
                .toString(16)
                .substring(1);
        }
        return s() + s() + s();
    })();

    csmapi.set_endpoint(sa.iottalk_url);

    if (mqtturl != undefined){
        profile['mqtt_enable'] = true;
    }

    if (profile.is_sim == undefined) profile.is_sim = false;
    if (profile.idf_list == undefined) profile.idf_list = [];
    if (profile.odf_list == undefined) profile.odf_list = [];	

    profile['df_list']=[];
    for (var i = 0; i < profile.odf_list.length; i++) {
        odf_name = profile.odf_list[i].name;
        if(odf_name[odf_name.length-2] == '_'){
            odf_name = odf_name.substr(0, odf_name.length-2) + '-' + odf_name.substr(odf_name.length-1);
        }
        odf_func[odf_name] = profile.odf_list[i];
		profile.odf_list[i] = odf_name;
		profile['df_list'].push(odf_name);
        console.log(odf_name);
    }
	
    for (var i = 0; i < profile.idf_list.length; i++) {
        idf_name = profile.idf_list[i].name;
        if(idf_name[idf_name.length-2] == '_'){
            idf_name = idf_name.substr(0, idf_name.length-2) + '-' + idf_name.substr(idf_name.length-1);
        }
        idf_func[idf_name] = profile.idf_list[i];
		profile.idf_list[i] = idf_name;
		profile['df_list'].push(idf_name);
        console.log(idf_name);
    }	

    function mqtt_publish(mac_addr, idf_name, data){
        if(!Array.isArray(data)) data = [data];
        topic = [mac_addr, idf_name].join('//');
        payload = {"samples":[[(moment().format('YYYY-MM-DD h:mm:ss')).toString(), data]]};
        sample = JSON.stringify(payload);
        mqtt_client.publish(topic, sample, {qos: 0, retain: false }, function (error){
            if(error) console.log(error);
            else console.log([topic, sample].join(': '));
        });
    }
	
    function push(idf_name, data=undefined){
        if (data == undefined) data = idf_func[idf_name]();
        if (data != undefined){
            if (mqtturl == undefined) dan.push(idf_name, data);
            else mqtt_publish(mac_addr, idf_name, data);
        }
	}
	
    function pull(odf_name, data) {
        if (odf_name == 'Control') {
            switch (data[0]) {
            case 'SET_DF_STATUS':
                dan.push('Control', ['SET_DF_STATUS_RSP', data[1]], function (res) {});
                break;
            case 'RESUME':
                sa.suspended = false;
                dan.push('Control', ['RESUME_RSP', ['OK']], function (res) {});
                break;
            case 'SUSPEND':
                sa.suspended = true;
                dan.push('Control', ['SUSPEND_RSP', ['OK']], function (res) {});
                break;
            }
        } else {
            odf_func[odf_name](data);
        }
    }


    if (mqtturl != undefined){
        const options = {
          clean: true, 
          connectTimeout: 4000, 
          clientId: mac_addr,
          username: user,
          password: password,
        }

        mqtt_client = mqtt.connect(mqtturl, options);

        mqtt_client.on('error', function(err){
            console.log('mqtt error:', err);
            mqtt_client.end();
            mqtt_client.reconnect();
        });

        mqtt_client.on('connect', function(connack){
            console.log('MQTT Broker connected.');
            profile.odf_list.forEach(function(odf){
                var topic = [mac_addr, odf].join('//');
                mqtt_client.subscribe(topic);
                console.log('subscribe:', topic);
            });
        });

        mqtt_client.on('message', function (topic, message, packet) {
            data = JSON.parse(message.toString());
            odf_name = topic.split('//')[1];
            odf_data = data['samples'][0][1];
            console.log(odf_name, odf_data);
            odf_func[odf_name](odf_data);
        });

        mqtt_client.on('disconnect', function (packet) {
            console.log(packet);
            mqtt_client.reconnect();
        });
    }


    function init_callback (result) {
        console.log('register:', result);
        document.title = profile.d_name;
        sa.sa_init(mac_addr);
    }


    function deregisterCallback (result) {
        console.log('deregister:', result);
    }

    function deregister () {
        dan.deregister(deregisterCallback);
    }

    window.onunload = deregister;
    window.onbeforeunload = deregister;
    window.onclose = deregister;
    window.onpagehide = deregister;
    
    dan.init(push, pull, csmapi.get_endpoint(), mac_addr, profile, init_callback, mqtturl, sa.exec_interval);

    function switch_to_http(){
        mqtt_client.end();
        mqtturl = undefined;
        profile['mqtt_enable'] = false;
        dan.init(push, pull, csmapi.get_endpoint(), mac_addr, profile, init_callback, undefined, sa.exec_interval);
    }
    http.init(switch_to_http);
};
