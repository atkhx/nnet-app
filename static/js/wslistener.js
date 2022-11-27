var wsListener = {
    wsClient: wsClient(),
    binds: {},
    bindsQueue: []
};

wsListener.isBinded = function (event, recipient) {
    return this.binds[event] !== undefined && this.binds[event][recipient] !== undefined;
};

wsListener.Bind = function (event, recipient, callback) {
    if (this.binds[event] === undefined) {
        this.binds[event] = {};
    }

    this.binds[event][recipient] = {
        callback: callback,
    };
};

wsListener.Unbind = function (event, recipient) {
    if (this.isBinded(event, recipient)) {
        delete this.binds[event][recipient];

        for (var i in this.binds[event]) {
            return;
        }

        this.wsClient.sendJSON({
            code: 'unsubscribe',
            data: {
                event: event
            }
        })
    }
};

wsListener.wsClient.onopen = function () {
    for (var e in wsListener.binds) {
        for (var r in wsListener.binds[e]) {
            wsListener.wsClient.sendJSON({
                code: 'subscribe',
                data: {
                    event: e,
                    receiver: r
                }
            })
        }
    }
};

wsListener.wsClient.onmessage = function (event) {
    var res = false;
    var obj = JSON.parse(event.data);

    switch (obj.code) {
        default:
            for (var r in wsListener.binds[obj.code]) {
                wsListener.binds[obj.code][r].callback(obj)
            }
    }
};

wsListener.wsClient.onerror = function (event) {
    // alert('error:' + JSON.stringify(event))
};
